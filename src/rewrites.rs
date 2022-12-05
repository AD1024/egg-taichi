use std::fmt::Display;

use crate::language::{ChiAnalysis, ChiIR, ConstData};
use egg::{rewrite as rw, Applier, RecExpr, Rewrite, Var};

type EGraph = egg::EGraph<ChiIR, ChiAnalysis>;

// fn is_pow_of_two() -> Fn(&mut EGraph, Id, &egg::Subst) -> bool {
//     |egraph, _,
// }

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
        searcher_ast: Option<&egg::PatternAst<ChiIR>>,
        rule_name: egg::Symbol,
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

pub fn alg_simp() -> Vec<Rewrite<ChiIR, ChiAnalysis>> {
    vec![
        rw!("smult-id"; "(smult 1 ?x)" => "?x"),
        rw!("smult-0"; "(smult 0 ?x)" => "0"),
        rw!("smult-comm"; "(smult ?x ?y)" => "(smult ?y ?x)"),
        rw!("smult-assoc"; "(smult (smult ?x ?y) ?z)" => "(smult ?x (smult ?y ?z))"),
        rw!("sadd-comm"; "(sadd ?x ?y)" => "(sadd ?y ?x)"),
        rw!("sadd-assoc"; "(sadd (sadd ?x ?y) ?z)" => "(sadd ?x (sadd ?y ?z))"),
        rw!("mult-dist"; "(smult ?x (sadd ?y ?z))" => "(sadd (smult ?x ?y) (smult ?x ?z))"),
        rw!("ewadd-comm"; "(ewadd ?x ?y)" => "(ewadd ?y ?x)"),
        rw!("ewadd-assoc"; "(ewadd (ewadd ?x ?y) ?z)" => "(ewadd ?x (ewadd ?y ?z))"),
        rw!("const-fold-add"; "(sadd ?x ?y)" => { BinopConstFoldApplier { lhs: "?x".parse().unwrap(), rhs: "?y".parse().unwrap(), op: "+".to_string() }}),
        rw!("const-fold-mult"; "(smult ?x ?y)" => { BinopConstFoldApplier { lhs: "?x".parse().unwrap(), rhs: "?y".parse().unwrap(), op: "*".to_string() }}),
        rw!("const-fold-div"; "(sdiv ?x ?y)" => { BinopConstFoldApplier { lhs: "?x".parse().unwrap(), rhs: "?y".parse().unwrap(), op: "/".to_string() }}),
        rw!("const-fold-sub"; "(sminus ?x ?y)" => { BinopConstFoldApplier { lhs: "?x".parse().unwrap(), rhs: "?y".parse().unwrap(), op: "-".to_string() }}),
        rw!("div-1"; "(sdiv ?x 1)" => "?x"),
        rw!("div-cast-1"; "(sdiv ?x 1)" => "?x"),
    ]
}

pub fn linalg_simp() -> Vec<Rewrite<ChiIR, ChiAnalysis>> {
    vec![
        rw!("matmul-assoc"; "(matmul (matmul ?x ?y) ?z)" <=> "(matmul ?x (matmul ?y ?z))"),
        rw!("transpose-matmul"; "(matmul (transpose ?x) (transpose ?y))" <=> "(transpose (matmul ?y ?x))"),
        vec![rw!("transpose-transpose"; "(transpose (transpose ?x))" => "?x")],
        rw!("matmul-bias"; "(matmul ?x (ewadd ?y ?z))" <=> "(ewadd (matmul ?x ?y) (matmul ?x ?z))"),
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
        runner
            .egraph
            .dot()
            .to_png("/root/test_const_fold.png")
            .unwrap();
    }
}
