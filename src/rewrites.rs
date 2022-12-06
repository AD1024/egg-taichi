use std::fmt::Display;

use crate::language::{AnalysisInfo, ChiAnalysis, ChiIR, ConstData, DataType};
use egg::{rewrite as rw, Applier, RecExpr, Rewrite, Var};

type EGraph = egg::EGraph<ChiIR, ChiAnalysis>;

fn is_scalar(x: Var) -> impl Fn(&mut EGraph, egg::Id, &egg::Subst) -> bool {
    move |egraph, _id, subst| match &egraph[subst[x]].data.analysis_info {
        AnalysisInfo::DType(dt) => match dt {
            DataType::Float(_) | DataType::Int(_) | DataType::UInt(_) => true,
            _ => false,
        },
        _ => false,
    }
}

struct BinopConstFoldApplier {
    lhs: Var,
    rhs: Var,
    op: String,
}

impl Applier<ChiIR, ChiAnalysis> for BinopConstFoldApplier {
    fn apply_one(
        &self,
        egraph: &mut egg::EGraph<ChiIR, ChiAnalysis>,
        eclass: egg::Id,
        subst: &egg::Subst,
        _: Option<&egg::PatternAst<ChiIR>>,
        _: egg::Symbol,
    ) -> Vec<egg::Id> {
        if let (Some(c1), Some(c2)) = (
            ChiAnalysis::get_constant(egraph, &subst[self.lhs]),
            ChiAnalysis::get_constant(egraph, &subst[self.rhs]),
        ) {
            fn add_const<T: Display>(egraph: &mut EGraph, eclass: &egg::Id, c: T) -> egg::Id {
                let new_id = egraph.add_expr(&format!("{}", c).parse::<RecExpr<_>>().unwrap());
                egraph.union(*eclass, new_id);
                new_id
            }
            if let Some(result) = match self.op.as_str() {
                "+" => Some(c1 + c2),
                "-" => Some(c1 - c2),
                "*" => Some(c1 * c2),
                "/" => match c2 {
                    ConstData::Int(0) => None,
                    _ => Some(c1 / c2),
                },
                "<<" => Some(ConstData::shl(&c1, &c2)),
                ">>" => Some(ConstData::shr(&c1, &c2)),
                "&" => Some(c1 & c2),
                "|" => Some(c1 | c2),
                _ => unimplemented!(),
            } {
                match result {
                    ConstData::Float(i) => {
                        vec![eclass, add_const(egraph, &eclass, i)]
                    }
                    ConstData::Bool(i) => {
                        vec![eclass, add_const(egraph, &eclass, i)]
                    }
                    ConstData::Int(i) => {
                        vec![eclass, add_const(egraph, &eclass, i)]
                    }
                    _ => unimplemented!(),
                }
            } else {
                vec![]
            }
        } else {
            vec![]
        }
    }
}

struct ToShiftApplier {
    rhs: Var,
    shift_op: &'static str,
}

impl Applier<ChiIR, ChiAnalysis> for ToShiftApplier {
    fn apply_one(
        &self,
        egraph: &mut egg::EGraph<ChiIR, ChiAnalysis>,
        eclass: egg::Id,
        subst: &egg::Subst,
        searcher_ast: Option<&egg::PatternAst<ChiIR>>,
        rule_name: egg::Symbol,
    ) -> Vec<egg::Id> {
        if let Some(denom) = ChiAnalysis::get_constant(egraph, &subst[self.rhs]) {
            if let ConstData::Int(denom) = denom {
                if denom & (denom - 1) == 0 {
                    let shift = denom.trailing_zeros();
                    let pattern = format!("({} ?lhs {})", self.shift_op, shift)
                        .parse::<egg::Pattern<ChiIR>>()
                        .unwrap();
                    return pattern.apply_one(egraph, eclass, subst, searcher_ast, rule_name);
                }
            }
        }
        return vec![];
    }
}

pub fn alg_simp() -> Vec<Rewrite<ChiIR, ChiAnalysis>> {
    vec![
        rw!("smult-id"; "(smult 1 ?x)" => "?x"),
        rw!("smult-0"; "(smult 0 ?x)" => "0"),
        rw!("smult-comm"; "(smult ?x ?y)" => "(smult ?y ?x)"),
        rw!("smult-assoc"; "(smult (smult ?x ?y) ?z)" => "(smult ?x (smult ?y ?z))"),
        rw!("sadd-comm"; "(sadd ?x ?y)" => "(sadd ?y ?x)"),
        rw!("sadd-assoc"; "(sadd (sadd ?x ?y) ?z)" => "(sadd ?x (sadd ?y ?z))"),
        rw!("mult-dist"; "(smult ?x (sadd ?y ?z))" => "(sadd (smult ?x ?y) (smult ?x ?z))"),
        rw!("mult-to-shift"; "(smult ?lhs ?x)" => { ToShiftApplier { rhs: "?x".parse().unwrap(), shift_op: "bitshl" } }),
        rw!("const-fold-add"; "(sadd ?x ?y)" => { BinopConstFoldApplier { lhs: "?x".parse().unwrap(), rhs: "?y".parse().unwrap(), op: "+".to_string() }}),
        rw!("const-fold-mult"; "(smult ?x ?y)" => { BinopConstFoldApplier { lhs: "?x".parse().unwrap(), rhs: "?y".parse().unwrap(), op: "*".to_string() }}),
        rw!("const-fold-div"; "(sdiv ?x ?y)" => { BinopConstFoldApplier { lhs: "?x".parse().unwrap(), rhs: "?y".parse().unwrap(), op: "/".to_string() }}),
        rw!("const-fold-sub"; "(sminus ?x ?y)" => { BinopConstFoldApplier { lhs: "?x".parse().unwrap(), rhs: "?y".parse().unwrap(), op: "-".to_string() }}),
        rw!("const-fold-bitshl"; "(bitshl ?x ?y)" => { BinopConstFoldApplier { lhs: "?x".parse().unwrap(), rhs: "?y".parse().unwrap(), op: "<<".to_string() }}),
        rw!("const-fold-bitshr"; "(bitshr ?x ?y)" => { BinopConstFoldApplier { lhs: "?x".parse().unwrap(), rhs: "?y".parse().unwrap(), op: ">>".to_string() }}),
        rw!("const-fold-bitand"; "(bitand ?x ?y)" => { BinopConstFoldApplier { lhs: "?x".parse().unwrap(), rhs: "?y".parse().unwrap(), op: "&".to_string() }}),
        rw!("const-fold-bitor"; "(bitor ?x ?y)" => { BinopConstFoldApplier { lhs: "?x".parse().unwrap(), rhs: "?y".parse().unwrap(), op: "|".to_string() }}),
        rw!("div-to-shift"; "(sdiv ?lhs ?x)" => { ToShiftApplier { rhs: "?x".parse().unwrap(), shift_op: "bitshr" }}),
        rw!("car-cons"; "(car (cons ?x ?y))" => "?x"),
        rw!("cdr-cons"; "(cdr (cons ?x ?y))" => "?y"),
        rw!("div-1"; "(sdiv ?x 1)" => "?x"),
        rw!("div-cast-1"; "(sdiv ?x 1)" => "?x"),
        rw!("ite-true"; "(ite true ?x ?y)" => "?x"),
        rw!("ite-false"; "(ite false ?x ?y)" => "?y"),
    ]
}

pub fn linalg_simp() -> Vec<Rewrite<ChiIR, ChiAnalysis>> {
    vec![
        rw!("matmul-assoc"; "(matmul (matmul ?x ?y) ?z)" <=> "(matmul ?x (matmul ?y ?z))"),
        rw!("transpose-matmul"; "(matmul (transpose ?x) (transpose ?y))" <=> "(transpose (matmul ?y ?x))"),
        rw!("matmul-bias"; "(matmul ?x (ewadd ?y ?z))" <=> "(ewadd (matmul ?x ?y) (matmul ?x ?z))"),
        rw!("transpose-transpose"; "(transpose (transpose ?x))" <=> "?x"),
        rw!("transpose-ewadd"; "(transpose (ewadd ?x ?y))" <=> "(ewadd (transpose ?x) (transpose ?y))"),
        rw!("smult-transpose"; "(smult (transpose ?x) ?w)" <=> "(transpose (smult ?x ?w))" if is_scalar("?w".parse().unwrap())),
        rw!("matmul-linear"; "(smult (matmul ?x ?y) ?w)" <=> "(matmul ?x (smult ?y ?w))" if is_scalar("?w".parse().unwrap())),
        rw!("ewadd-comm"; "(ewadd ?x ?y)" <=> "(ewadd ?y ?x)"),
        rw!("ewadd-assoc"; "(ewadd (ewadd ?x ?y) ?z)" <=> "(ewadd ?x (ewadd ?y ?z))"),
        rw!("ewmult-comm"; "(ewmult ?x ?y)" <=> "(ewmult ?y ?x)"),
        rw!("ewmult-assoc"; "(ewmult (ewmult ?x ?y) ?z)" <=> "(ewmult ?x (ewmult ?y ?z))"),
        rw!("ewmult-dist"; "(ewmult ?x (ewadd ?y ?z))" <=> "(ewadd (ewmult ?x ?y) (ewmult ?x ?z))"),
        rw!("smult-ewadd-dist"; "(smult (ewadd ?x ?y) ?w)" <=> "(ewadd (smult ?x ?w) (smult ?y ?w))" if is_scalar("?w".parse().unwrap())),
    ].concat()
}

mod tests {
    use egg::RecExpr;

    use crate::language::ChiAnalysis;

    #[test]
    fn test_const_fold() {
        let expr = "(smult i (sadd x N))".parse::<RecExpr<_>>().unwrap();
        let mut egraph = egg::EGraph::new(ChiAnalysis::default());
        let _ = egraph.add_expr(&expr);
        let runner = egg::Runner::default()
            .with_expr(&expr)
            .with_egraph(egraph)
            .run(&crate::rewrites::alg_simp());
        // runner
        //     .egraph
        //     .dot()
        //     .to_png("/root/test_const_fold.png")
        //     .unwrap();
    }
}
