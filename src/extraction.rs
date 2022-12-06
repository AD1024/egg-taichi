use egg::{CostFunction, EGraph, Language};

use crate::language::{ChiAnalysis, ChiIR};

struct ComputeCost<'a> {
    egraph: &'a EGraph<ChiIR, ChiAnalysis>,
}

impl<'a> CostFunction<ChiIR> for ComputeCost<'a> {
    type Cost = i64;

    fn cost<C>(&mut self, enode: &ChiIR, mut costs: C) -> Self::Cost
    where
        C: FnMut(egg::Id) -> Self::Cost,
    {
        let c = match enode {
            ChiIR::Constant(_) | ChiIR::Var(_) | ChiIR::Matrix(_) | ChiIR::Vector(_) => 1,
            ChiIR::SAdd([x, y])
            | ChiIR::SMinus([x, y])
            | ChiIR::SMult([x, y])
            | ChiIR::SMod([x, y])
            | ChiIR::BitAnd([x, y])
            | ChiIR::BitOr([x, y])
            | ChiIR::BitXor([x, y])
            | ChiIR::BitShl([x, y])
            | ChiIR::BitShr([x, y])
            | ChiIR::Equals([x, y])
            | ChiIR::Gt([x, y])
            | ChiIR::Lt([x, y])
            | ChiIR::Gte([x, y])
            | ChiIR::Lte([x, y])
            | ChiIR::SDiv([x, y]) => costs(*x) + costs(*y) + 1,
            ChiIR::EWAdd([x, _]) | ChiIR::EWMult([x, _]) => costs(*x) * 2,
            ChiIR::Seq([s1, s2]) => costs(*s1) + costs(*s2),
            ChiIR::While([cond, body]) => costs(*cond) + costs(*body),
            ChiIR::IfThenElse([cond, then, els]) => costs(*cond) + costs(*then) + costs(*els),
            ChiIR::Transpose([x]) => costs(*x),
            ChiIR::MatMul([x, y]) => {
                let x_shape = ChiAnalysis::get_shape(self.egraph, x);
                let y_shape = ChiAnalysis::get_shape(self.egraph, y);
                (x_shape[0] * x_shape[1] * y_shape[1] + 1) as i64
            }
            ChiIR::Cons([x, y]) => {
                let x_cost = costs(*x);
                let y_cost = costs(*y);
                x_cost + y_cost + 1
            }
            ChiIR::Car([x]) => match &self.egraph[*x].nodes[0] {
                ChiIR::Cons([car, _]) => costs(*car) + 1,
                _ => panic!(
                    "Expecting a list construction, got {:?>}",
                    self.egraph[*x].nodes[0]
                ),
            },
            ChiIR::Cdr([x]) => match &self.egraph[*x].nodes[0] {
                ChiIR::Cons([_, cdr]) => costs(*cdr) + 1,
                _ => panic!(
                    "Expecting a list construction, got {:?>}",
                    self.egraph[*x].nodes[0]
                ),
            },
            ChiIR::Symbol(_) => 1,
            ChiIR::Nil => 0,
            _ => unimplemented!("Cost function not implemented for {:?}", enode),
        };
        enode.fold(c, |acc, id| acc + costs(id))
    }
}

mod test {
    use std::collections::HashMap;

    use crate::{
        language::{ChiAnalysis, ChiIR, ConstData, DataType},
        rewrites::alg_simp,
        rewrites::linalg_simp,
    };

    #[test]
    fn test_cse() {
        let expr: egg::RecExpr<ChiIR> = "(cons
                                            (sadd j (smult i (sadd N 1)))
                                            (cons
                                                (sadd 1 (sadd j (smult i (sadd N 1))))
                                                (sadd N (sadd 2 (sadd j (smult i (sadd N 1))))))
                                        )"
        .parse()
        .unwrap();
        let mut egraph: egg::EGraph<ChiIR, ChiAnalysis> = egg::EGraph::new(ChiAnalysis {
            constants: [("N".to_string(), 16.into())].iter().cloned().collect(),
            name_to_shapes: Default::default(),
            name_to_type: [
                ("i".to_string(), DataType::Int(32)),
                ("j".to_string(), DataType::Int(32)),
                ("N".to_string(), DataType::Int(32)),
            ]
            .iter()
            .cloned()
            .collect(),
        });
        let rt = egraph.add_expr(&expr);
        let runner = egg::Runner::default().with_egraph(egraph);
        let runner = runner.run(&alg_simp());
        let extractor = egg::Extractor::new(
            &runner.egraph,
            super::ComputeCost {
                egraph: &runner.egraph,
            },
        );
        let (cost, expr) = extractor.find_best(rt);
        println!("Cost: {}", cost);
        println!("Expr: {}", expr.pretty(80));
        // runner.egraph.dot().to_png("/root/cse.png").unwrap();
        // println!("{:?}", runner.egraph.dump());
    }

    /// A simple MLP model
    /// Output = (((input * W1 + b1) * W2 + b2) * W3 + b3)
    #[test]
    fn test_mlp() {
        let expr: egg::RecExpr<ChiIR> =
            "(ewadd (matmul (ewadd (matmul (ewadd (matmul input W1) b1) W2) b2) W3) b3)"
                .parse()
                .unwrap();
        let mut egraph: egg::EGraph<ChiIR, ChiAnalysis> = egg::EGraph::new(ChiAnalysis {
            constants: HashMap::default(),
            name_to_shapes: [
                ("input".to_string(), vec![32, 32]),
                ("W1".to_string(), vec![32, 64]),
                ("b1".to_string(), vec![32, 64]),
                ("W2".to_string(), vec![64, 128]),
                ("b2".to_string(), vec![32, 128]),
                ("W3".to_string(), vec![128, 10]),
                ("b3".to_string(), vec![32, 10]),
            ]
            .iter()
            .cloned()
            .collect(),
            name_to_type: [
                ("input".to_string(), DataType::Float(32)),
                ("W1".to_string(), DataType::Float(32)),
                ("b1".to_string(), DataType::Float(32)),
                ("W2".to_string(), DataType::Float(32)),
                ("b2".to_string(), DataType::Float(32)),
                ("W3".to_string(), DataType::Float(32)),
                ("b3".to_string(), DataType::Float(32)),
            ]
            .iter()
            .cloned()
            .collect(),
        });
        let rt = egraph.add_expr(&expr);
        let runner = egg::Runner::default().with_egraph(egraph);
        let extractor = egg::Extractor::new(
            &runner.egraph,
            super::ComputeCost {
                egraph: &runner.egraph,
            },
        );
        let (cost, _) = extractor.find_best(rt);
        println!("Cost before optimization: {}", cost);
        drop(extractor);
        let mut rules = alg_simp();
        rules.extend(linalg_simp());
        let runner = egg::Runner::default()
            .with_egraph(runner.egraph)
            .run(&rules);
        let extractor = egg::Extractor::new(
            &runner.egraph,
            super::ComputeCost {
                egraph: &runner.egraph,
            },
        );
        let (cost, expr) = extractor.find_best(rt);
        println!("Cost after optimization: {}", cost);
        println!("Expr: {}", expr.pretty(80));
    }
}
