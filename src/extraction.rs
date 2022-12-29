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
            ChiIR::Constant(_)
            | ChiIR::Var(_)
            | ChiIR::Matrix(_)
            | ChiIR::Vector(_)
            | ChiIR::Index(_) => 1,
            ChiIR::SAdd([x, y]) | ChiIR::SMinus([x, y]) => 3,
            ChiIR::SMult([x, y]) => 8,
            ChiIR::SMod([x, y]) | ChiIR::SDiv([x, y]) => 20,
            ChiIR::BitAnd([x, y])
            | ChiIR::BitOr([x, y])
            | ChiIR::BitXor([x, y])
            | ChiIR::BitShl([x, y])
            | ChiIR::BitShr([x, y]) => 1,
            ChiIR::Equals([x, y])
            | ChiIR::Gt([x, y])
            | ChiIR::Lt([x, y])
            | ChiIR::Gte([x, y])
            | ChiIR::Lte([x, y]) => i64::max(costs(*x), costs(*y)),
            ChiIR::EWAdd([x, _]) => ChiAnalysis::get_shape(self.egraph, x)
                .iter()
                .map(|x| *x as i64)
                .sum::<i64>(),
            ChiIR::EWMult([x, _]) => {
                2 * ChiAnalysis::get_shape(self.egraph, x)
                    .iter()
                    .map(|x| *x as i64)
                    .sum::<i64>()
            }
            ChiIR::Seq([_, _]) => 0,
            ChiIR::While([_, _]) => 0,
            ChiIR::IfThenElse([_, _, _]) => 0,
            ChiIR::Transpose([x]) => ChiAnalysis::get_shape(self.egraph, x)
                .iter()
                .map(|x| *x as i64)
                .sum::<i64>(),
            ChiIR::MatMul([x, y]) => {
                let x_shape = ChiAnalysis::get_shape(self.egraph, x);
                let y_shape = ChiAnalysis::get_shape(self.egraph, y);
                (x_shape[0] * x_shape[1] * y_shape[1]) as i64
            }
            ChiIR::Cons([_, _]) => 1,
            ChiIR::Car([x]) => match &self.egraph[*x].nodes[0] {
                ChiIR::Cons([_, _]) => 1,
                _ => panic!(
                    "Expecting a list construction, got {:?>}",
                    self.egraph[*x].nodes[0]
                ),
            },
            ChiIR::Cdr([x]) => match &self.egraph[*x].nodes[0] {
                ChiIR::Cons([_, _]) => 1,
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
                                            (smult (sadd (smult i N) j) 2)
                                                (cons
                                                    (sadd j (smult i (sadd N 1)))
                                                    (cons
                                                        (sadd 1 (sadd j (smult i (sadd N 1))))
                                                        (sadd N (sadd 2 (sadd j (smult i (sadd N 1))))))))"
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
        let extractor = egg::Extractor::new(
            &runner.egraph,
            super::ComputeCost {
                egraph: &runner.egraph,
            },
        );
        println!("Cost before optimization: {}", extractor.find_best(rt).0);
        drop(extractor);
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
                ("input".to_string(), vec![16, 16]),
                ("W1".to_string(), vec![16, 32]),
                ("b1".to_string(), vec![16, 32]),
                ("W2".to_string(), vec![32, 64]),
                ("b2".to_string(), vec![16, 64]),
                ("W3".to_string(), vec![64, 10]),
                ("b3".to_string(), vec![16, 10]),
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
        println!("Expr: {}", expr.pretty(80));
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
