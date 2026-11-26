//! Two console lanes for each host thread.
//!
//! A console wave64 compute program whose lanes read one another needs waves
//! of 64 lanes; a host whose waves are 32 lanes at most (NVIDIA) refuses that.
//! The forms the lowering has for such a host cut a console wave into two host
//! waves that meet at group barriers, which is exact only where every thread
//! of the group gets to the barrier, and a card's wait for a thread that never
//! comes ends in a device reset.
//!
//! This pass takes the module the lowering makes for a host of 64-lane waves
//! (the form AMD cards run) and makes every thread do the work of two lanes:
//! host lane `l` of a wave runs the console's lanes `l` and `l + 32`. One host
//! wave of 32 threads then holds a whole console wave, so what the lanes ask
//! of one another is answered by the host's own wave operations: no barrier,
//! nothing shared, nothing to wait for. The thread group is half the
//! console's, in one row (`Module::pair` leaves the declaration of that to its
//! caller).
//!
//! How:
//! - Code every lane of the wave runs together stays in one piece. Each of
//!   its instructions whose value or effect is a lane's own is made twice,
//!   once for either lane, in the order the console code has them (the lanes
//!   of a wave run an instruction together; here the lower lane is first).
//!   What is the same for the whole wave is made once.
//! - A wave operation is made over both lanes: a ballot is the lower lanes'
//!   bits and then the upper lanes', a lane read picks the half the lane is
//!   in, a reduction joins the two lanes first, a prefix adds the lower
//!   half's total to the upper lanes'.
//! - Where a branch goes by a lane's own condition (a guard around one
//!   instruction, a loop of one lane's own), the two lanes of a thread may go
//!   different ways. Such a region ("zone": the blocks between the branch and
//!   where its ways meet again) is made twice, and the thread runs the lower
//!   lane's copy and then the upper lane's. A zone must not hold a wave
//!   operation or a barrier (the lanes are not together there): the program
//!   is then refused, and its caller has the lowering take those branches
//!   wave-wide.
//!
//! The lanes of a console wave are the host lane index (and that plus 32),
//! and a thread's index in its group is taken from them, as the lowering does
//! for console lanes. A group of several console waves relies on the host
//! giving a wave a 32-aligned run of thread indices of the halved group.

use std::collections::HashMap;

use super::opt::{Effect, effect};
use super::{Bin, Block, Inst, Module, Op, Term, Ty, TypeDef, V, pred, successors};

/// Why a module cannot be paired.
pub(crate) type Refusal = &'static str;

/// A wave operation or a barrier where a thread's two lanes are apart.
pub(crate) const APART: Refusal = "a wave operation where a thread's two lanes are apart";

/// One lane's own loop with the translation's limit of iterations.
pub(crate) const LIMIT: Refusal = "a lane's own loop with a limit of iterations";

/// The `dx.op` opcodes the pass knows by number.
mod code {
    pub const BARRIER: i64 = 80;
    pub const THREAD_ID: i64 = 93;
    pub const GROUP_ID: i64 = 94;
    pub const THREAD_ID_IN_GROUP: i64 = 95;
    pub const FLAT_THREAD: i64 = 96;
    pub const IS_FIRST_LANE: i64 = 110;
    pub const LANE_INDEX: i64 = 111;
    pub const LANE_COUNT: i64 = 112;
    pub const ANY_TRUE: i64 = 113;
    pub const ALL_TRUE: i64 = 114;
    pub const BALLOT: i64 = 116;
    pub const READ_LANE_AT: i64 = 117;
    pub const READ_LANE_FIRST: i64 = 118;
    pub const ACTIVE_OP: i64 = 119;
    pub const ACTIVE_BIT: i64 = 120;
    pub const PREFIX_OP: i64 = 121;
}

/// Where a zone's lanes go and meet.
struct Zone {
    /// The block outside that leads in (its closing branch is each lane's).
    decider: usize,
    /// The one block the zone leaves to.
    exit: usize,
    /// The new blocks where the lower and the upper lane's copies end.
    joins: [usize; 2],
}

struct Pass<'m> {
    m: &'m mut Module,
    abort: Option<usize>,
    insts: Vec<Inst>,
    blocks: Vec<Block>,
    preds: Vec<Vec<usize>>,
    /// Per old instruction: its value differs between a thread's two lanes,
    /// or it is an effect each lane has.
    varying: Vec<bool>,
    /// Per old block: its closing branch goes by a lane's own condition.
    lane_branch: Vec<bool>,
    /// Per old block: the zone it is in.
    zone_of: Vec<Option<usize>>,
    zones: Vec<Zone>,
    /// Per old block: the zone it decides.
    decides: Vec<Option<usize>>,
    /// Per lane, per old instruction: the new value.
    val: [Vec<Option<V>>; 2],
    /// Per old ballot: the lower and the upper lanes' ballots.
    ballots: HashMap<u32, [V; 2]>,
    /// Per old block outside the zones: its new block; per old block in a
    /// zone: either lane's copy.
    new: Vec<[usize; 2]>,
    /// New phis to fill: (new phi, lane, old phi, what kind of place).
    phis: Vec<(V, usize, u32, Place)>,
    /// Per zone, old phi of its exit and lane: the phi where the lane's copy ends.
    joined: HashMap<(usize, u32, usize), V>,
}

#[derive(Clone, Copy)]
enum Place {
    /// A phi of a block outside the zones, or of a zone's block.
    Block(usize),
    /// A zone's exit's phi, joined where the lane's copy of the zone ends.
    Join(usize),
}

impl Module {
    /// See the module's documentation. On a refusal the module is as it was.
    ///
    /// `abort`: the block a thread whose loop ran to the translation's limit of
    /// iterations leaves through. A loop the wave runs together leaves by it together,
    /// as before; one lane's own loop has no such way (its thread's other lane goes on),
    /// and a program with one is refused.
    pub(crate) fn pair(&mut self, abort: Option<usize>) -> Result<(), Refusal> {
        let insts = std::mem::take(&mut self.insts);
        let blocks = std::mem::take(&mut self.blocks);
        let n = insts.len();
        let nb = blocks.len();
        let mut pass = Pass {
            m: self,
            abort,
            insts,
            blocks,
            preds: vec![Vec::new(); nb],
            varying: vec![false; n],
            lane_branch: vec![false; nb],
            zone_of: vec![None; nb],
            zones: Vec::new(),
            decides: vec![None; nb],
            val: [vec![None; n], vec![None; n]],
            ballots: HashMap::new(),
            new: vec![[usize::MAX; 2]; nb],
            phis: Vec::new(),
            joined: HashMap::new(),
        };
        let r = pass.run();
        if r.is_err() {
            // as it was (what the pass added of constants and declarations stays: unused)
            let Pass { m, insts, blocks, .. } = pass;
            m.insts = insts;
            m.blocks = blocks;
        } else {
            self.tags.clear();
            self.together.clear();
        }
        r
    }
}

impl Pass<'_> {
    fn run(&mut self) -> Result<(), Refusal> {
        let order = self.order();
        for &b in &order {
            for &s in successors(self.blocks[b].term).iter() {
                self.preds[s].push(b);
            }
        }
        let after = self.meets(&order)?;
        self.analyse(&order, &after)?;
        self.find_zones(&order)?;
        self.emit(&order)
    }

    /// The reached blocks in reverse postorder.
    fn order(&self) -> Vec<usize> {
        let n = self.blocks.len();
        let mut seen = vec![false; n];
        let mut post = Vec::new();
        if n == 0 {
            return post;
        }
        let mut stack = vec![(0usize, 0usize)];
        seen[0] = true;
        while let Some(top) = stack.last_mut() {
            let (b, k) = *top;
            top.1 += 1;
            match successors(self.blocks[b].term).get(k) {
                Some(&s) if !seen[s] => {
                    seen[s] = true;
                    stack.push((s, 0));
                }
                Some(_) => {}
                None => {
                    post.push(b);
                    stack.pop();
                }
            }
        }
        post.reverse();
        post
    }

    /// Per block: where every way on from it meets again (its immediate
    /// post-dominator; `usize::MAX`: the function's end).
    fn meets(&self, order: &[usize]) -> Result<Vec<usize>, Refusal> {
        let n = self.blocks.len();
        const END: usize = usize::MAX;
        // Postorder of the reversed graph from the end: a block's number is higher the
        // nearer the end it is.
        let mut number = vec![usize::MAX; n];
        let rpo = {
            let mut seen = vec![false; n];
            let mut post = Vec::new();
            let roots: Vec<usize> =
                order
                    .iter()
                    .copied()
                    .filter(|&b| matches!(self.blocks[b].term, Some(Term::Ret)) && Some(b) != self.abort)
                    .collect();
            for root in roots {
                if seen[root] {
                    continue;
                }
                seen[root] = true;
                let mut stack = vec![(root, 0usize)];
                while let Some(top) = stack.last_mut() {
                    let (b, k) = *top;
                    top.1 += 1;
                    match self.preds[b].get(k) {
                        Some(&p) if !seen[p] && Some(b) != self.abort => {
                            seen[p] = true;
                            stack.push((p, 0));
                        }
                        Some(_) => {}
                        None => {
                            post.push(b);
                            stack.pop();
                        }
                    }
                }
            }
            post.reverse();
            for (k, &b) in post.iter().enumerate() {
                number[b] = k + 1;
            }
            post
        };
        if order.iter().any(|&b| number[b] == usize::MAX && Some(b) != self.abort) {
            return Err("a block that never reaches the program's end");
        }
        // (the end is number 0)
        let num = |b: usize| if b == END { 0 } else { number[b] };
        let mut ipdom = vec![usize::MAX - 1; n];
        const NONE: usize = usize::MAX - 1;
        let mut changed = true;
        while changed {
            changed = false;
            for &b in &rpo {
                let mut new = NONE;
                let ways: Vec<usize> = match self.blocks[b].term {
                    Some(Term::Ret) | None => vec![END],
                    _ => self.ways(b),
                };
                for s in ways {
                    if s != END && ipdom[s] == NONE {
                        continue;
                    }
                    if new == NONE {
                        new = s;
                        continue;
                    }
                    let (mut x, mut y) = (s, new);
                    while x != y {
                        while num(x) > num(y) {
                            x = ipdom[x];
                        }
                        while num(y) > num(x) {
                            y = ipdom[y];
                        }
                    }
                    new = x;
                }
                if new != NONE && ipdom[b] != new {
                    ipdom[b] = new;
                    changed = true;
                }
            }
        }
        if order.iter().any(|&b| ipdom[b] == NONE && Some(b) != self.abort) {
            return Err("a block whose ways on do not meet");
        }
        Ok(ipdom)
    }

    /// Where block `b` goes on to, the way out at the loop limit left aside.
    fn ways(&self, b: usize) -> Vec<usize> {
        successors(self.blocks[b].term).iter().copied().filter(|&s| Some(s) != self.abort).collect()
    }

    fn opcode(&self, i: u32) -> Option<i64> {
        match &self.insts[i as usize].op {
            Op::Call(f, args) if self.m.funcs[*f as usize].name.starts_with("dx.op.") => {
                args.first().and_then(|&a| self.m.int_value(a))
            }
            _ => None,
        }
    }

    /// (Diagnostics.) `v` as an expression, `depth` levels deep.
    fn show(&self, v: V, depth: u32) -> String {
        match v {
            V::Inst(i) => {
                let inst = &self.insts[i as usize];
                let mark = if self.varying[i as usize] { "~" } else { "" };
                if depth == 0 {
                    return format!("{mark}%{i}");
                }
                let mut ops = Vec::new();
                super::for_each_operand(&inst.op, |o| ops.push(self.show(o, depth - 1)));
                let name = match &inst.op {
                    Op::Call(f, _) => self.m.funcs[*f as usize].name.trim_start_matches("dx.op.").to_owned(),
                    Op::Bin(b, ..) => format!("{b:?}"),
                    Op::Cmp(p, ..) => format!("cmp{p}"),
                    other => format!("{other:?}").split('(').next().unwrap_or("").to_owned(),
                };
                format!("{mark}%{i}={name}({})", ops.join(", "))
            }
            V::Const(_) => match self.m.int_value(v) {
                Some(k) => format!("{k}"),
                None => "k".to_owned(),
            },
            other => format!("{other:?}"),
        }
    }

    fn var(&self, v: V) -> bool {
        matches!(v, V::Inst(i) if self.varying[i as usize])
    }

    /// Which values are a lane's own, and which branches go by them; the
    /// blocks between such a branch and where its ways meet are a zone's.
    fn analyse(&mut self, order: &[usize], after: &[usize]) -> Result<(), Refusal> {
        let mut in_zone = vec![false; self.blocks.len()];
        // Phis that join what the lanes of a zone did.
        let mut joined = vec![false; self.insts.len()];
        loop {
            let mut changed = false;
            for &b in order {
                let ids: Vec<u32> = self.blocks[b].phis.iter().chain(&self.blocks[b].body).copied().collect();
                for i in ids {
                    if self.varying[i as usize] {
                        continue;
                    }
                    let v = in_zone[b] || joined[i as usize] || self.own(i)?;
                    if v {
                        self.varying[i as usize] = true;
                        changed = true;
                    }
                }
                // (a branch the lowering knows the wave takes together is no lane's own,
                // whatever its condition was made of: every thread makes the same of it)
                if let Some(Term::CondBr(c, ..)) = self.blocks[b].term
                    && self.var(c)
                    && !self.lane_branch[b]
                    && !self.m.together.contains(&b)
                {
                    self.lane_branch[b] = true;
                    changed = true;
                    let meet = after[b];
                    if std::env::var_os("VICTORY_PAIR_DEBUG").is_some_and(|v| v == "2") {
                        let mut text = String::new();
                        let mut v = c;
                        for _ in 0..14 {
                            let V::Inst(i) = v else { break };
                            let op = &self.insts[i as usize].op;
                            let name = match op {
                                Op::Call(f, _) => self.m.funcs[*f as usize].name.clone(),
                                other => format!("{other:?}").split('(').next().unwrap_or("").to_owned(),
                            };
                            text.push_str(&name);
                            text.push_str(" <- ");
                            let mut next = None;
                            super::for_each_operand(op, |o| {
                                if next.is_none() && self.var(o) {
                                    next = Some(o);
                                }
                            });
                            match next {
                                Some(n) => v = n,
                                None => break,
                            }
                        }
                        if std::env::var_os("VICTORY_PAIR_SHOW").is_some() && text.contains("waveReadLaneAt") {
                            eprintln!("PAIR-SHOW {}", self.show(c, 9));
                        }
                        eprintln!(
                            "PAIR-LANE-BRANCH block {b} zone {} ways {:?} meet {} : {text}",
                            in_zone[b],
                            successors(self.blocks[b].term).to_vec(),
                            meet as isize
                        );
                    }
                    if meet == usize::MAX {
                        if std::env::var_os("VICTORY_PAIR_DEBUG").is_some() {
                            let describe = |v: V| match v {
                                V::Inst(i) => format!("{:?}", self.insts[i as usize].op),
                                other => format!("{other:?}"),
                            };
                            let mut text = describe(c);
                            if let V::Inst(i) = c {
                                super::for_each_operand(&self.insts[i as usize].op, |v| {
                                    text.push_str(" <- ");
                                    text.push_str(&describe(v));
                                });
                            }
                            let ways: Vec<String> = successors(self.blocks[b].term)
                                .iter()
                                .map(|&s| {
                                    format!(
                                        "{s}:{}insts,{:?}",
                                        self.blocks[s].body.len() + self.blocks[s].phis.len(),
                                        self.blocks[s].term
                                    )
                                })
                                .collect();
                            eprintln!("PAIR-END block {b} of {} cond {text} ways {ways:?}", self.blocks.len());
                        }
                        return Err("a lane's own way to the program's end");
                    }
                    // what lies between the branch and the meeting
                    let mut stack: Vec<usize> = self.ways(b);
                    while let Some(x) = stack.pop() {
                        if x == meet || in_zone[x] {
                            continue;
                        }
                        in_zone[x] = true;
                        stack.extend(self.ways(x));
                    }
                    for &phi in &self.blocks[meet].phis {
                        joined[phi as usize] = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        for (b, &z) in in_zone.iter().enumerate() {
            if z {
                // (numbered when the zones are found)
                self.zone_of[b] = Some(usize::MAX);
            }
        }
        Ok(())
    }

    /// Whether instruction `i` outside the zones is a lane's own, by what it
    /// is and what it reads.
    fn own(&self, i: u32) -> Result<bool, Refusal> {
        let inst = &self.insts[i as usize];
        let mut reads = false;
        super::for_each_operand(&inst.op, |v| reads |= self.var(v));
        Ok(match &inst.op {
            Op::AtomicRmw(..) | Op::CmpXchg(..) => true,
            Op::Call(f, args) => match self.opcode(i) {
                Some(code::LANE_INDEX | code::IS_FIRST_LANE | code::PREFIX_OP) => true,
                Some(
                    code::ANY_TRUE
                    | code::ALL_TRUE
                    | code::BALLOT
                    | code::READ_LANE_FIRST
                    | code::ACTIVE_OP
                    | code::ACTIVE_BIT
                    | code::LANE_COUNT
                    | code::FLAT_THREAD
                    | code::GROUP_ID
                    | code::BARRIER,
                ) => false,
                // the value at a lane: the same for both where they ask for the same lane
                Some(code::READ_LANE_AT) => args.get(2).is_some_and(|&l| self.var(l)),
                Some(code::THREAD_ID | code::THREAD_ID_IN_GROUP) => {
                    return Err("a thread's id taken from the host");
                }
                _ => match effect(&self.m.funcs[*f as usize]) {
                    Effect::Lanes => return Err("a wave operation the pairing does not know"),
                    // each lane's own store, atomic or output
                    Effect::Writes => true,
                    Effect::Pure | Effect::Reads => reads,
                },
            },
            _ => reads,
        })
    }

    /// The zones: what a block outside leads into, with the one block it
    /// comes out at.
    fn find_zones(&mut self, order: &[usize]) -> Result<(), Refusal> {
        let effects = self.m.effects();
        for &q in order {
            if self.zone_of[q].is_some() {
                continue;
            }
            let into: Vec<usize> = self.ways(q).into_iter().filter(|&s| self.zone_of[s].is_some()).collect();
            if into.is_empty() {
                continue;
            }
            let z = self.zones.len();
            let mut members = Vec::new();
            let mut exit = None;
            let mut stack = into;
            while let Some(x) = stack.pop() {
                match self.zone_of[x] {
                    Some(usize::MAX) => {}
                    Some(other) if other == z => continue,
                    Some(_) => return Err("a lane's own region that two places lead into"),
                    None => {
                        if exit.is_some_and(|e| e != x) {
                            return Err("a lane's own region with two ways out");
                        }
                        exit = Some(x);
                        continue;
                    }
                }
                self.zone_of[x] = Some(z);
                members.push(x);
                match self.blocks[x].term {
                    Some(Term::Ret) | None => return Err("a lane's own way to the program's end"),
                    t if successors(t).iter().any(|&s| Some(s) == self.abort) => return Err(LIMIT),
                    t => stack.extend(successors(t).iter().copied()),
                }
            }
            let Some(exit) = exit else {
                return Err("a lane's own region without a way out");
            };
            // The decider's own ways: into the zone, to its exit, or (a branch the wave
            // takes together) somewhere else altogether.
            if self.lane_branch[q] && successors(self.blocks[q].term).iter().any(|&s| self.zone_of[s] != Some(z) && s != exit)
            {
                return Err("a lane's own branch that leaves its region");
            }
            // Nothing in it that the lanes do together, and no way in but the decider's.
            let mut cyclic = false;
            let mut shared = false;
            for &x in &members {
                for &p in &self.preds[x] {
                    if p != q && self.zone_of[p] != Some(z) {
                        return Err("a lane's own region that two places lead into");
                    }
                }
                for &i in self.blocks[x].phis.iter().chain(&self.blocks[x].body) {
                    match &self.insts[i as usize].op {
                        Op::Call(f, _) => match self.opcode(i) {
                            Some(code::LANE_INDEX | code::FLAT_THREAD | code::GROUP_ID | code::LANE_COUNT) => {}
                            Some(code::BARRIER) => return Err(APART),
                            Some(code::THREAD_ID | code::THREAD_ID_IN_GROUP) => {
                                return Err("a thread's id taken from the host");
                            }
                            _ if effects[*f as usize] == Effect::Lanes => return Err(APART),
                            _ => {}
                        },
                        Op::Load(p) | Op::Store(p, _) | Op::AtomicRmw(_, p, _) | Op::CmpXchg(p, ..) => {
                            shared |= self.shared_memory(*p);
                        }
                        _ => {}
                    }
                }
                cyclic |= self.reaches_itself(x, z);
            }
            // The lanes of a wave run an instruction together: a loop of one lane's own
            // over memory the group shares would let the lower lane run all its turns
            // before the upper lane's first.
            if cyclic && shared {
                return Err(APART);
            }
            self.decides[q] = Some(z);
            self.zones.push(Zone { decider: q, exit, joins: [usize::MAX; 2] });
        }
        if self.zone_of.iter().any(|z| *z == Some(usize::MAX)) {
            return Err("a lane's own region nothing leads into");
        }
        Ok(())
    }

    /// Whether `p` points into the group's shared memory.
    fn shared_memory(&self, p: V) -> bool {
        match p {
            V::Global(g) => self.m.globals[g as usize].init.is_none(),
            V::Inst(i) => match &self.insts[i as usize].op {
                Op::Gep(_, base, _) => self.shared_memory(*base),
                _ => true,
            },
            _ => false,
        }
    }

    fn reaches_itself(&self, x: usize, z: usize) -> bool {
        let mut seen = vec![false; self.blocks.len()];
        let mut stack: Vec<usize> = successors(self.blocks[x].term).to_vec();
        while let Some(b) = stack.pop() {
            if b == x {
                return true;
            }
            if self.zone_of[b] != Some(z) || std::mem::replace(&mut seen[b], true) {
                continue;
            }
            stack.extend(successors(self.blocks[b].term).iter().copied());
        }
        false
    }

    fn emit(&mut self, order: &[usize]) -> Result<(), Refusal> {
        // The entry stays the first block.
        for &b in order {
            if self.zone_of[b].is_none() {
                let k = self.m.new_block();
                self.new[b] = [k, k];
            } else {
                self.new[b] = [self.m.new_block(), self.m.new_block()];
            }
        }
        for z in 0..self.zones.len() {
            self.zones[z].joins = [self.m.new_block(), self.m.new_block()];
        }
        for &b in order {
            match self.zone_of[b] {
                None => self.together(b)?,
                Some(_) => {
                    for lane in 0..2 {
                        self.alone(b, lane)?;
                    }
                }
            }
        }
        self.fill_phis();
        Ok(())
    }

    /// The new value of `v` for `lane`.
    fn get(&self, v: V, lane: usize) -> V {
        match v {
            V::Inst(i) => self.val[lane][i as usize].expect("made before it is read"),
            other => other,
        }
    }

    fn set(&mut self, i: u32, a: V, b: V) {
        self.val[0][i as usize] = Some(a);
        self.val[1][i as usize] = Some(b);
    }

    fn push(&mut self, block: usize, ty: Ty, op: Op) -> V {
        self.m.push(block, ty, op)
    }

    /// `old` once more for `lane`, in new block `block`.
    fn copy(&mut self, old: u32, lane: usize, block: usize) -> V {
        let Inst { ty, mut op } = self.insts[old as usize].clone();
        let vals = &self.val[lane];
        let fix = |v: V| match v {
            V::Inst(i) => vals[i as usize].expect("made before it is read"),
            other => other,
        };
        match &mut op {
            Op::Bin(_, a, b) | Op::Cmp(_, a, b) | Op::Store(a, b) | Op::AtomicRmw(_, a, b) => {
                *a = fix(*a);
                *b = fix(*b);
            }
            Op::Cast(_, v) | Op::Extract(v, _) | Op::Load(v) => *v = fix(*v),
            Op::Select(a, b, c) | Op::CmpXchg(a, b, c) => {
                *a = fix(*a);
                *b = fix(*b);
                *c = fix(*c);
            }
            Op::Call(_, args) => args.iter_mut().for_each(|a| *a = fix(*a)),
            Op::Gep(_, p, idx) => {
                *p = fix(*p);
                idx.iter_mut().for_each(|i| *i = fix(*i));
            }
            Op::Phi(_) => unreachable!("phis are made apart"),
        }
        self.push(block, ty, op)
    }

    fn new_phi(&mut self, block: usize, old: u32, lane: usize, place: Place) -> V {
        let ty = self.insts[old as usize].ty;
        let phi = self.m.phi(block, ty);
        self.phis.push((phi, lane, old, place));
        phi
    }

    /// A block every lane of the wave runs together.
    fn together(&mut self, b: usize) -> Result<(), Refusal> {
        let at = self.new[b][0];
        // Whether it is where a zone's lanes come out.
        let exit_of: Vec<usize> = (0..self.zones.len()).filter(|&z| self.zones[z].exit == b).collect();
        for &phi in &self.blocks[b].phis.clone() {
            if self.varying[phi as usize] || !exit_of.is_empty() {
                let a = self.new_phi(at, phi, 0, Place::Block(b));
                let c = self.new_phi(at, phi, 1, Place::Block(b));
                self.set(phi, a, c);
            } else {
                let a = self.new_phi(at, phi, 0, Place::Block(b));
                self.set(phi, a, a);
            }
        }
        // (a zone's exit's phis are joined for either lane where its copy ends)
        for &z in &exit_of {
            for &phi in &self.blocks[b].phis.clone() {
                for lane in 0..2 {
                    let join = self.zones[z].joins[lane];
                    let v = self.new_phi(join, phi, lane, Place::Join(z));
                    self.joined.insert((z, phi, lane), v);
                }
            }
        }
        for &i in &self.blocks[b].body.clone() {
            self.instruction(i, at)?;
        }
        self.close(b)
    }

    /// One instruction of a block the lanes run together.
    fn instruction(&mut self, i: u32, at: usize) -> Result<(), Refusal> {
        let inst = self.insts[i as usize].clone();
        let i1t = self.m.int(1);
        if let Op::Extract(V::Inst(src), field) = inst.op
            && let Some(&[lower, upper]) = self.ballots.get(&src)
        {
            // a ballot's first dword is lanes 0 to 31, its second 32 to 63
            let v = match field {
                0 => self.m.extract(at, lower, 0),
                1 => self.m.extract(at, upper, 0),
                _ => self.m.i32(0),
            };
            self.set(i, v, v);
            return Ok(());
        }
        let Op::Call(f, ref args) = inst.op else {
            return self.plain(i, at);
        };
        let Some(op) = self.opcode(i) else {
            return self.plain(i, at);
        };
        let arg = |p: &Self, k: usize, lane: usize| p.get(args[k], lane);
        match op {
            code::LANE_INDEX => {
                let lower = self.push(at, inst.ty, inst.op.clone());
                let k32 = self.m.i32(32);
                let upper = self.m.bin(at, Bin::Add, lower, k32);
                self.set(i, lower, upper);
            }
            code::IS_FIRST_LANE => {
                let lower = self.push(at, inst.ty, inst.op.clone());
                let no = self.m.bool(false);
                self.set(i, lower, no);
            }
            code::LANE_COUNT => {
                let v = self.m.i32(64);
                self.set(i, v, v);
            }
            code::FLAT_THREAD => {
                // the console thread index its wave begins at: twice the 32-aligned run of
                // the halved group's indices the host wave holds
                let flat = self.push(at, inst.ty, inst.op.clone());
                let five = self.m.i32(5);
                let six = self.m.i32(6);
                let wave = self.m.bin(at, Bin::LShr, flat, five);
                let base = self.m.bin(at, Bin::Shl, wave, six);
                self.set(i, base, base);
            }
            code::ANY_TRUE | code::ALL_TRUE => {
                let (a, b) = (arg(self, 1, 0), arg(self, 1, 1));
                let both = if a == b {
                    a
                } else {
                    self.m.bin(at, if op == code::ANY_TRUE { Bin::Or } else { Bin::And }, a, b)
                };
                let v = self.push(at, i1t, Op::Call(f, vec![args[0], both]));
                self.set(i, v, v);
            }
            code::BALLOT => {
                let (a, b) = (arg(self, 1, 0), arg(self, 1, 1));
                let lower = self.push(at, inst.ty, Op::Call(f, vec![args[0], a]));
                let upper = if a == b { lower } else { self.push(at, inst.ty, Op::Call(f, vec![args[0], b])) };
                self.ballots.insert(i, [lower, upper]);
            }
            code::READ_LANE_FIRST => {
                // the wave's first lane is the first thread's lower one
                let a = arg(self, 1, 0);
                let v = self.push(at, inst.ty, Op::Call(f, vec![args[0], a]));
                self.set(i, v, v);
            }
            code::READ_LANE_AT => {
                let (a, b) = (arg(self, 1, 0), arg(self, 1, 1));
                let mut out = [a; 2];
                for lane in 0..2 {
                    if lane == 1 && !self.var(args[2]) {
                        out[1] = out[0];
                        break;
                    }
                    let index = arg(self, 2, lane);
                    out[lane] = match self.m.int_value(index) {
                        Some(k) => {
                            let k31 = self.m.i32(k as u32 & 31);
                            let from = if k as u32 & 32 == 0 { a } else { b };
                            self.push(at, inst.ty, Op::Call(f, vec![args[0], from, k31]))
                        }
                        None => {
                            let k31 = self.m.i32(31);
                            let k32 = self.m.i32(32);
                            let low = self.m.bin(at, Bin::And, index, k31);
                            let ra = self.push(at, inst.ty, Op::Call(f, vec![args[0], a, low]));
                            if a == b {
                                ra
                            } else {
                                let rb = self.push(at, inst.ty, Op::Call(f, vec![args[0], b, low]));
                                let bit = self.m.bin(at, Bin::And, index, k32);
                                let zero = self.m.i32(0);
                                let upper = self.m.cmp(at, pred::NE, bit, zero);
                                self.m.select(at, upper, rb, ra)
                            }
                        }
                    };
                }
                self.set(i, out[0], out[1]);
            }
            code::ACTIVE_OP => {
                let (a, b) = (arg(self, 1, 0), arg(self, 1, 1));
                let kind = self.m.int_value(args[2]).ok_or("a wave reduction of an unknown kind")?;
                let unsigned = self.m.int_value(args[3]).ok_or("a wave reduction of an unknown kind")? != 0;
                let both = self.join(at, kind, unsigned, a, b)?;
                let v = self.push(at, inst.ty, Op::Call(f, vec![args[0], both, args[2], args[3]]));
                self.set(i, v, v);
            }
            code::ACTIVE_BIT => {
                let (a, b) = (arg(self, 1, 0), arg(self, 1, 1));
                let both = match self.m.int_value(args[2]) {
                    _ if a == b && self.m.int_value(args[2]) != Some(2) => a,
                    Some(0) => self.m.bin(at, Bin::And, a, b),
                    Some(1) => self.m.bin(at, Bin::Or, a, b),
                    Some(2) => self.m.bin(at, Bin::Xor, a, b),
                    _ => return Err("a wave reduction of an unknown kind"),
                };
                let v = self.push(at, inst.ty, Op::Call(f, vec![args[0], both, args[2]]));
                self.set(i, v, v);
            }
            code::PREFIX_OP => {
                // the lower lanes' prefix is theirs; the upper lanes' begins at the lower
                // lanes' total
                if self.m.int_value(args[2]) != Some(0) {
                    return Err("a wave prefix of an operation other than a sum");
                }
                let (a, b) = (arg(self, 1, 0), arg(self, 1, 1));
                let lower = self.push(at, inst.ty, Op::Call(f, vec![args[0], a, args[2], args[3]]));
                let total = self.total(at, inst.ty, a, args[3]);
                let upper = self.push(at, inst.ty, Op::Call(f, vec![args[0], b, args[2], args[3]]));
                let upper = self.m.bin(at, Bin::Add, total, upper);
                self.set(i, lower, upper);
            }
            code::THREAD_ID | code::THREAD_ID_IN_GROUP => return Err("a thread's id taken from the host"),
            _ if effect(&self.m.funcs[f as usize]) == Effect::Lanes => {
                return Err("a wave operation the pairing does not know");
            }
            _ => return self.plain(i, at),
        }
        Ok(())
    }

    /// The sum over the wave of `v` (the lower lanes' values), by the host's own reduction.
    fn total(&mut self, at: usize, ty: Ty, v: V, unsigned: V) -> V {
        let name = match self.m.types[ty.0 as usize] {
            TypeDef::Float => "dx.op.waveActiveOp.f32",
            TypeDef::Double => "dx.op.waveActiveOp.f64",
            TypeDef::Int(64) => "dx.op.waveActiveOp.i64",
            _ => "dx.op.waveActiveOp.i32",
        };
        let (i32t, i8t) = (self.m.int(32), self.m.int(8));
        let fty = self.m.func_ty(ty, vec![i32t, ty, i8t, i8t]);
        self.m.ptr(fty);
        let f = self.m.function(name, fty, Some(super::Attr::NoUnwind), false);
        let V::Func(f) = f else { unreachable!() };
        let code = self.m.i32(code::ACTIVE_OP as u32);
        let sum = self.m.i8(0);
        self.push(at, ty, Op::Call(f, vec![code, v, sum, unsigned]))
    }

    /// What a reduction of kind `kind` makes of a thread's two lanes.
    fn join(&mut self, at: usize, kind: i64, unsigned: bool, a: V, b: V) -> Result<V, Refusal> {
        let ty = self.m.type_of(a);
        let float = matches!(self.m.types[ty.0 as usize], TypeDef::Float | TypeDef::Double);
        Ok(match kind {
            0 => self.m.bin(at, Bin::Add, a, b),
            1 => self.m.bin(at, Bin::Mul, a, b),
            2 | 3 if a == b => a,
            2 | 3 => {
                let less = kind == 2;
                if float {
                    // the other where one is not a number, as the host's reduction leaves
                    // those out
                    let p = if less { pred::OLT } else { pred::OGT };
                    let first = self.m.cmp(at, p, a, b);
                    let t = self.m.select(at, first, a, b);
                    let nan = self.m.cmp(at, pred::UNO, b, b);
                    self.m.select(at, nan, a, t)
                } else {
                    let p = match (less, unsigned) {
                        (true, true) => pred::IULT,
                        (true, false) => pred::SLT,
                        (false, true) => pred::IUGT,
                        (false, false) => pred::SGT,
                    };
                    let first = self.m.cmp(at, p, a, b);
                    self.m.select(at, first, a, b)
                }
            }
            _ => return Err("a wave reduction of an unknown kind"),
        })
    }

    /// An instruction that is no wave operation: once if the wave's, twice if the lanes'.
    fn plain(&mut self, i: u32, at: usize) -> Result<(), Refusal> {
        let mut ballot = false;
        super::for_each_operand(&self.insts[i as usize].op, |v| {
            ballot |= matches!(v, V::Inst(s) if self.ballots.contains_key(&s));
        });
        if ballot {
            return Err("a ballot used whole");
        }
        let a = self.copy(i, 0, at);
        let b = if self.varying[i as usize] { self.copy(i, 1, at) } else { a };
        self.set(i, a, b);
        Ok(())
    }

    /// The branch that closes block `b`, which the lanes run together.
    fn close(&mut self, b: usize) -> Result<(), Refusal> {
        let at = self.new[b][0];
        let term = self.blocks[b].term;
        let Some(z) = self.decides[b] else {
            let t = match term {
                Some(Term::Br(s)) => Term::Br(self.new[s][0]),
                Some(Term::CondBr(c, s, t)) => Term::CondBr(self.get(c, 0), self.new[s][0], self.new[t][0]),
                Some(Term::Ret) | None => Term::Ret,
            };
            self.m.terminate(at, t);
            return Ok(());
        };
        // Into a zone: the lower lane's way from here, the upper lane's from where the
        // lower lane's copy ends, and on from where the upper's does.
        let (exit, joins) = (self.zones[z].exit, self.zones[z].joins);
        for lane in 0..2 {
            let from = if lane == 0 { at } else { joins[0] };
            let to = |p: &Self, s: usize| {
                if p.zone_of[s] == Some(z) {
                    Some(p.new[s][lane])
                } else if s == exit {
                    Some(joins[lane])
                } else {
                    None
                }
            };
            let t = match term {
                Some(Term::Br(s)) => Term::Br(to(self, s).expect("the zone's")),
                Some(Term::CondBr(c, s, t)) => match (to(self, s), to(self, t)) {
                    (Some(s), Some(t)) => Term::CondBr(self.get(c, lane), s, t),
                    // a way the wave takes together past the zone: the lower lane's
                    // branch decides it for both
                    (Some(s), None) if lane == 0 => Term::CondBr(self.get(c, 0), s, self.new[t][0]),
                    (None, Some(t)) if lane == 0 => Term::CondBr(self.get(c, 0), self.new[s][0], t),
                    (Some(s), None) | (None, Some(s)) => Term::Br(s),
                    (None, None) => unreachable!("a decider leads into its zone"),
                },
                Some(Term::Ret) | None => unreachable!("a decider branches"),
            };
            self.m.terminate(from, t);
        }
        self.m.terminate(joins[1], Term::Br(self.new[exit][0]));
        Ok(())
    }

    /// A zone's block, for one lane.
    fn alone(&mut self, b: usize, lane: usize) -> Result<(), Refusal> {
        let at = self.new[b][lane];
        for &phi in &self.blocks[b].phis.clone() {
            let v = self.new_phi(at, phi, lane, Place::Block(b));
            self.val[lane][phi as usize] = Some(v);
        }
        for &i in &self.blocks[b].body.clone() {
            let inst = self.insts[i as usize].clone();
            let v = match self.opcode(i) {
                Some(code::LANE_INDEX) => {
                    let l = self.push(at, inst.ty, inst.op.clone());
                    if lane == 0 {
                        l
                    } else {
                        let k32 = self.m.i32(32);
                        self.m.bin(at, Bin::Add, l, k32)
                    }
                }
                Some(code::LANE_COUNT) => self.m.i32(64),
                Some(code::FLAT_THREAD) => {
                    let flat = self.push(at, inst.ty, inst.op.clone());
                    let five = self.m.i32(5);
                    let six = self.m.i32(6);
                    let wave = self.m.bin(at, Bin::LShr, flat, five);
                    self.m.bin(at, Bin::Shl, wave, six)
                }
                _ => {
                    let mut ballot = false;
                    super::for_each_operand(&inst.op, |v| {
                        ballot |= matches!(v, V::Inst(s) if self.ballots.contains_key(&s));
                    });
                    if let Op::Extract(V::Inst(src), field) = inst.op
                        && let Some(&[lower, upper]) = self.ballots.get(&src)
                    {
                        match field {
                            0 => self.m.extract(at, lower, 0),
                            1 => self.m.extract(at, upper, 0),
                            _ => self.m.i32(0),
                        }
                    } else if ballot {
                        return Err("a ballot used whole");
                    } else {
                        self.copy(i, lane, at)
                    }
                }
            };
            self.val[lane][i as usize] = Some(v);
        }
        let z = self.zone_of[b].expect("a zone's block");
        let (exit, join) = (self.zones[z].exit, self.zones[z].joins[lane]);
        let to = |p: &Self, s: usize| if s == exit { join } else { p.new[s][lane] };
        let t = match self.blocks[b].term {
            Some(Term::Br(s)) => Term::Br(to(self, s)),
            Some(Term::CondBr(c, s, t)) => Term::CondBr(self.get(c, lane), to(self, s), to(self, t)),
            Some(Term::Ret) | None => unreachable!("refused before"),
        };
        self.m.terminate(at, t);
        Ok(())
    }

    fn fill_phis(&mut self) {
        for (phi, lane, old, place) in std::mem::take(&mut self.phis) {
            let Op::Phi(incoming) = self.insts[old as usize].op.clone() else { unreachable!() };
            let mut new = Vec::new();
            match place {
                Place::Block(b) => {
                    let inside = self.zone_of[b];
                    // the zones this block is the exit of, joined already
                    let mut done: Vec<usize> = Vec::new();
                    for (v, p) in incoming {
                        if self.new[p][0] == usize::MAX {
                            // (from a block nothing reaches)
                            continue;
                        }
                        match (inside, self.zone_of[p], self.decides[p]) {
                            // inside a zone, from its own blocks
                            (Some(z), Some(pz), _) if z == pz => new.push((self.get(v, lane), self.new[p][lane])),
                            // inside a zone, from its decider: the lower lane comes from
                            // there, the upper from where the lower's copy ended
                            (Some(z), None, _) => {
                                let from = if lane == 0 { self.new[p][0] } else { self.zones[z].joins[0] };
                                new.push((self.get(v, lane), from));
                            }
                            (Some(_), Some(_), _) => unreachable!("zones do not lead into one another"),
                            // out of a zone, or past it from its decider: through the joins
                            (None, Some(z), _) => {
                                if !done.contains(&z) {
                                    done.push(z);
                                    new.push((self.joined[&(z, old, lane)], self.zones[z].joins[1]));
                                }
                            }
                            (None, None, Some(z)) if self.zones[z].exit == b => {
                                // straight from the decider to the exit: through the joins
                                // too (either lane's branch is made where its turn begins)
                                if !done.contains(&z) {
                                    done.push(z);
                                    new.push((self.joined[&(z, old, lane)], self.zones[z].joins[1]));
                                }
                            }
                            (None, None, _) => new.push((self.get(v, lane), self.new[p][0])),
                        }
                    }
                }
                Place::Join(z) => {
                    let decider = self.zones[z].decider;
                    for (v, p) in incoming {
                        if self.new[p][0] == usize::MAX {
                            continue;
                        }
                        if self.zone_of[p] == Some(z) {
                            new.push((self.get(v, lane), self.new[p][lane]));
                        } else if p == decider {
                            // straight from the decider: the lower lane from it, the upper
                            // from where the lower's copy ended
                            let from = if lane == 0 { self.new[p][0] } else { self.zones[z].joins[0] };
                            new.push((self.get(v, lane), from));
                        }
                    }
                }
            }
            self.m.set_phi(phi, new);
        }
    }
}
