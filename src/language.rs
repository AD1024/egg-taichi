use egg::{self, define_language, Analysis, DidMerge, Id};
use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    ops::Deref,
    str::FromStr,
};

define_language! {
    enum ChiIR {
        // Scala addition
        // (sadd s1 s2)
        "sadd" = SAdd([Id; 2]),
        "sminus" = SMinus([Id; 2]),
        "smult" = SMult([Id; 2]),
        "sdiv" = SDiv([Id; 2]),

        ">=" = Gte([Id; 2]),
        "<=" = Lte([Id; 2]),
        ">" = Gt([Id; 2]),
        "<" = Lt([Id; 2]),
        "==" = Equals([Id; 2]),
        // Vector init
        // (vector <list> <data-type>)
        "vector" = Vector(Vec<Id>),
        // Matrix init
        // (matrix <list of vectors> <data-type>)
        "matrix" = Matrix(Vec<Id>),
        "matmul" = MatMul([Id; 2]),
        // element-wise addition
        // (ewadd mat/vec mat/vec)
        "ewadd" = EWAdd([Id; 2]),
        // Matrix transpose
        "transpose" = Transpose([Id; 1]),
        "concat" = Concat([Id; 2]),

        "ite" = IfThenElse([Id; 3]),
        // Suffix binding
        // (compute <expr> <bindings>)
        "compute" = Compute(Vec<Id>),
        // Let binding; use with compute
        "let" = Let([Id; 2]),
        "var" = Var(Id),
        "literal" = Literal(Vec<Id>),
        "while" = While([Id; 2]),
        "shape" = Shape(Vec<Id>),
        DataType(DataType),
        Symbol(String),
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
enum DataType {
    Int(usize),
    Float(usize),
    Bool,
    UInt(usize),
    TensorType(Box<DataType>, Vec<usize>),
    Unknown,
}

impl Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataType::Int(32) => write!(f, "i32"),
            DataType::Int(64) => write!(f, "i64"),
            DataType::Float(32) => write!(f, "f32"),
            DataType::Float(64) => write!(f, "f64"),
            DataType::Float(16) => write!(f, "f16"),
            DataType::Bool => write!(f, "bool"),
            DataType::UInt(n) => write!(f, "u{}", n),
            DataType::TensorType(t, s) => write!(f, "{}{:?}", t, s),
            DataType::Unknown => write!(f, "unknown"),
            _ => panic!("Unknown data type: {:?}", self),
        }
    }
}

impl FromStr for DataType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "i32" => Ok(DataType::Int(32)),
            "i64" => Ok(DataType::Int(64)),
            "f32" => Ok(DataType::Float(32)),
            "f64" => Ok(DataType::Float(64)),
            "f16" => Ok(DataType::Float(16)),
            "bool" => Ok(DataType::Bool),
            "u32" => Ok(DataType::UInt(32)),
            "u64" => Ok(DataType::UInt(64)),
            "unknown" => Ok(DataType::Unknown),
            _ => Err(format!("Unknown data type: {}", s)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChiAnalysis {
    pub state: HashMap<String, Id>,
    pub name_to_shapes: HashMap<String, Vec<usize>>,
}

#[derive(Debug, Clone)]
enum ChiAnalysisData {
    DType(DataType),
    Shape(Vec<usize>),
    LoopVar(HashMap<ChiIR, HashSet<String>>),
}

fn promote_dtype(x: &DataType, y: &DataType) -> DataType {
    match x {
        DataType::Unknown => y.clone(),
        DataType::Int(x_bits) => match y {
            DataType::Unknown => x.clone(),
            DataType::Int(y_bits) => {
                if x_bits > y_bits {
                    x.clone()
                } else {
                    y.clone()
                }
            }
            DataType::UInt(_) => y.clone(),
            DataType::Float(_) => y.clone(),
            DataType::Bool => panic!("Cannot decide dtype between bool and int"),
            DataType::TensorType(_, _) => panic!("Cannot decide dtype between tensor and int"),
        },
        DataType::UInt(x_bits) => match y {
            DataType::Unknown => x.clone(),
            DataType::Int(_) => x.clone(),
            DataType::UInt(y_bits) => {
                if x_bits > y_bits {
                    x.clone()
                } else {
                    y.clone()
                }
            }
            DataType::Float(_) => y.clone(),
            DataType::Bool => panic!("Cannot decide dtype between bool and uint"),
            DataType::TensorType(_, _) => panic!("Cannot decide dtype between tensor and uint"),
        },
        DataType::Float(x_bits) => match y {
            DataType::Unknown => x.clone(),
            DataType::Int(_) => x.clone(),
            DataType::UInt(_) => x.clone(),
            DataType::Float(y_bits) => {
                if x_bits > y_bits {
                    x.clone()
                } else {
                    y.clone()
                }
            }
            DataType::Bool => panic!("Cannot decide dtype between bool and float"),
            DataType::TensorType(_, _) => panic!("Cannot decide dtype between tensor and float"),
        },
        DataType::Bool => match y {
            DataType::Bool => x.clone(),
            _ => panic!("Cannot decide dtype between bool and non-bool"),
        },
        DataType::TensorType(x_dtype, x_shape) => match y {
            DataType::TensorType(y_dtype, y_shape) => {
                if y_shape != x_shape {
                    panic!("Cannot decide dtype between tensors with different shapes");
                }
                DataType::TensorType(Box::new(promote_dtype(x_dtype, y_dtype)), x_shape.clone())
            }
            _ => panic!("Cannot decide dtype between tensor and non-tensor"),
        },
    }
}

impl Analysis<ChiIR> for ChiAnalysis {
    type Data = ChiAnalysisData;

    fn merge(&mut self, lhs: &mut ChiAnalysisData, rhs: ChiAnalysisData) -> DidMerge {
        match (lhs, &rhs) {
            (ChiAnalysisData::DType(l), ChiAnalysisData::DType(r)) => {
                if l == r {
                    DidMerge(false, false)
                } else {
                    panic!("Type mismatch: {:?} vs {:?}", l, r);
                }
            }
            (ChiAnalysisData::Shape(l), ChiAnalysisData::Shape(r)) => {
                if l == r {
                    DidMerge(false, false)
                } else {
                    panic!("Shape mismatch: {:?} vs {:?}", l, r);
                }
            }
            (ChiAnalysisData::LoopVar(l), ChiAnalysisData::LoopVar(r)) => {
                let mut modified = false;
                for (k, v) in r {
                    if l.contains_key(&k) {
                        let loop_vars = l.get(&k).unwrap();
                        for var in v {
                            if !loop_vars.contains(var) {
                                panic!("Loop variable mismatch: {:?} vs {:?}", loop_vars, v);
                            }
                        }
                    } else {
                        l.insert(k.clone(), v.clone());
                        modified = true;
                    }
                }
                if !modified {
                    DidMerge(false, false)
                } else {
                    DidMerge(true, true)
                }
            }
            (lhs, rhs) => panic!("Cannot merge {:?} and {:?}", lhs, rhs),
        }
    }

    fn make(egraph: &egg::EGraph<ChiIR, Self>, enode: &ChiIR) -> ChiAnalysisData {
        match enode {
            ChiIR::SMinus([x, y])
            | ChiIR::Gt([x, y])
            | ChiIR::Lt([x, y])
            | ChiIR::Equals([x, y])
            | ChiIR::Gte([x, y])
            | ChiIR::Lte([x, y])
            | ChiIR::SMult([x, y])
            | ChiIR::SDiv([x, y])
            | ChiIR::SAdd([x, y]) => match (&egraph[*x].data, &egraph[*y].data) {
                (ChiAnalysisData::DType(x_dtype), ChiAnalysisData::DType(y_dtype)) => {
                    let dtype = promote_dtype(&x_dtype, &y_dtype);
                    ChiAnalysisData::DType(dtype)
                }
                _ => panic!("Unexpected dtype"),
            },
            ChiIR::Matrix(m) => {
                assert!(m.len() > 1);
                let dt = &egraph[*m.last().unwrap()].data;
                let decl_type = if let ChiAnalysisData::DType(dt) = dt {
                    dt.clone()
                } else {
                    DataType::Unknown
                };
                let dtype = &egraph[m[0]].data;
                if let ChiAnalysisData::DType(dt) = dtype {
                    if let DataType::TensorType(dtype, v_shape) = dt {
                        let row_len = v_shape[0];
                        let mut final_dtype = dtype.deref().clone();
                        m.iter()
                            .take(m.len() - 1)
                            .for_each(|x| match &egraph[*x].data {
                                ChiAnalysisData::DType(dt) => {
                                    if let DataType::TensorType(t_type, t_shape) = dt {
                                        if t_shape != v_shape {
                                            panic!("Shape mismatch for constructing matrix");
                                        }
                                        final_dtype = promote_dtype(&final_dtype, t_type);
                                    } else {
                                        panic!("Unexpected dtype");
                                    }
                                }
                                _ => panic!("Unexpected dtype"),
                            });
                        return ChiAnalysisData::DType(DataType::TensorType(
                            Box::new(if decl_type == DataType::Unknown { final_dtype } else { decl_type }),
                            vec![m.len() - 1, row_len],
                        ));
                    }
                }
                panic!("DType error for constructing matrix: {:?}", dtype);
            }
            ChiIR::Vector(v) => {
                assert!(v.len() > 1);
                let dt = &egraph[*v.last().unwrap()].data;
                let mut decl_dtype = if let ChiAnalysisData::DType(dt) = dt {
                    dt.clone()
                } else {
                    DataType::Unknown
                };
                v.iter().take(v.len() - 1).for_each(|x| {
                    if let ChiAnalysisData::DType(dt) = &egraph[*x].data {
                        decl_dtype = promote_dtype(&decl_dtype, dt);
                    } else {
                        panic!("Unexpected dtype");
                    }
                });
                ChiAnalysisData::DType(DataType::TensorType(
                    Box::new(decl_dtype),
                    vec![v.len() - 1],
                ))
            }
            ChiIR::Literal(x) => {
                if x.len() == 1 {
                    ChiAnalysisData::DType(DataType::Unknown)
                } else {
                    if let ChiAnalysisData::DType(dt) = &egraph[*x.last().unwrap()].data {
                        ChiAnalysisData::DType(dt.clone())
                    } else {
                        panic!("Unexpected dtype: {:?}", egraph[*x.last().unwrap()].data);
                    }
                }
            }
            _ => unimplemented!()
        }
    }
}

mod test {
    use super::*;
    use egg::RecExpr;
    #[test]
    fn test_parse_language() {
        let x = "(transpose (matrix (vector 1 2 3) (vector 4 5 6) (vector 7 8 9) i32))"
            .parse::<RecExpr<ChiIR>>()
            .unwrap();
        println!("{:?}", x);
        let x = "(compute (ewadd y z)
                                   (let y (vector (literal 1.0 f32) (literal 2.0) (literal 3.0)))
                                   (let z (vector (literal 5.0) (literal 6.0) (literal 7.0))))"
            .parse::<RecExpr<ChiIR>>()
            .unwrap();
        println!("{:?}", x);
    }
}
