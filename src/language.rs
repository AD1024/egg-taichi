use egg::{self, define_language, Analysis, DidMerge, EGraph, Id};
use ordered_float::NotNan;
use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    ops::Deref,
    str::FromStr,
};

define_language! {
    pub enum ChiIR {
        // Scala addition
        // (sadd s1 s2)
        "sadd" = SAdd([Id; 2]),
        "sminus" = SMinus([Id; 2]),
        "smult" = SMult([Id; 2]),
        "sdiv" = SDiv([Id; 2]),
        "smod" = SMod([Id; 2]),

        ">=" = Gte([Id; 2]),
        "<=" = Lte([Id; 2]),
        ">" = Gt([Id; 2]),
        "<" = Lt([Id; 2]),
        "==" = Equals([Id; 2]),

        "bitand" = BitAnd([Id; 2]),
        "bitor" = BitOr([Id; 2]),
        "bitxor" = BitXor([Id; 2]),
        "bitnot" = BitNot([Id; 1]),
        "bitshl" = BitShl([Id; 2]),
        "bitshr" = BitShr([Id; 2]),
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
        "load" = Load([Id; 1]),
        // (store <src> <dst>)
        "store" = Store([Id; 2]),
        "var" = Var(Id),
        "cast" = Cast([Id; 2]),
        "while" = While([Id; 2]),
        "shape" = Shape(Vec<Id>),
        DataType(DataType),
        Constant(NotNan<f64>),
        Symbol(String),
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum DataType {
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
pub enum ChiAnalysisData {
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
            | ChiIR::SMult([x, y])
            | ChiIR::SDiv([x, y])
            | ChiIR::SMod([x, y])
            | ChiIR::SAdd([x, y]) => match (&egraph[*x].data, &egraph[*y].data) {
                (ChiAnalysisData::DType(x_dtype), ChiAnalysisData::DType(y_dtype)) => {
                    if let DataType::TensorType(dt, _) = x_dtype {
                        return ChiAnalysisData::DType(promote_dtype(dt, y_dtype));
                    } else if let DataType::TensorType(dt, _) = y_dtype {
                        return ChiAnalysisData::DType(promote_dtype(x_dtype, dt));
                    }
                    let dtype = promote_dtype(x_dtype, y_dtype);
                    ChiAnalysisData::DType(dtype)
                }
                _ => panic!("Unexpected dtype"),
            },
            ChiIR::Gt([_x, _y])
            | ChiIR::Lt([_x, _y])
            | ChiIR::Equals([_x, _y])
            | ChiIR::Gte([_x, _y])
            | ChiIR::Lte([_x, _y]) => ChiAnalysisData::DType(DataType::Bool),
            ChiIR::BitAnd([x, y])
            | ChiIR::BitOr([x, y])
            | ChiIR::BitXor([x, y])
            | ChiIR::BitShl([x, y])
            | ChiIR::BitShr([x, y]) => match (&egraph[*x].data, &egraph[*y].data) {
                (ChiAnalysisData::DType(x_dtype), ChiAnalysisData::DType(y_dtype)) => {
                    let dtype = promote_dtype(&x_dtype, &y_dtype);
                    if let DataType::Int(_) = dtype {
                        ChiAnalysisData::DType(dtype)
                    } else {
                        panic!("Unexpected dtype: {:?}", dtype);
                    }
                }
                _ => panic!(
                    "Unexpected data for bit operation: {:?} vs {:?}",
                    &egraph[*x].data, &egraph[*y].data
                ),
            },
            ChiIR::BitNot([x]) => match &egraph[*x].data {
                ChiAnalysisData::DType(x_dtype) => {
                    if let DataType::Int(_) = x_dtype {
                        ChiAnalysisData::DType(x_dtype.clone())
                    } else {
                        panic!("Unexpected dtype: {:?}", x_dtype);
                    }
                }
                _ => panic!("Unexpected data for bit operation: {:?}", &egraph[*x].data),
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
                        let mut row_cnt = 0;
                        let mut final_dtype = dtype.deref().clone();
                        m.iter().take(m.len()).for_each(|x| match &egraph[*x].data {
                            ChiAnalysisData::DType(dt) => {
                                if let DataType::TensorType(t_type, t_shape) = dt {
                                    if t_shape != v_shape {
                                        panic!("Shape mismatch for constructing matrix");
                                    }
                                    final_dtype = promote_dtype(&final_dtype, t_type);
                                    row_cnt += 1;
                                }
                            }
                            _ => panic!("Unexpected data during matrix init: {:?}", dtype),
                        });
                        // TODO: need to check the declared type is compatible with the inferred type
                        return ChiAnalysisData::DType(DataType::TensorType(
                            Box::new(if decl_type == DataType::Unknown {
                                final_dtype
                            } else {
                                decl_type
                            }),
                            vec![row_cnt, v_shape[0]],
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
                let mut elem_count = 0;
                v.iter().take(v.len()).for_each(|x| {
                    if let ChiAnalysisData::DType(dt) = &egraph[*x].data {
                        decl_dtype = promote_dtype(&decl_dtype, dt);
                        elem_count += 1;
                    }
                });
                ChiAnalysisData::DType(DataType::TensorType(Box::new(decl_dtype), vec![elem_count]))
            }
            ChiIR::DataType(dtype) => ChiAnalysisData::DType(dtype.clone()),
            ChiIR::Cast([x, t]) => {
                if let (ChiAnalysisData::DType(x_dt), ChiAnalysisData::DType(dst_ty)) =
                    (&egraph[*x].data, &egraph[*t].data)
                {
                    if let DataType::TensorType(_x_elem_type, shape) = x_dt {
                        // TODO: check whether can be cast
                        return ChiAnalysisData::DType(DataType::TensorType(
                            Box::new(dst_ty.clone()),
                            shape.clone(),
                        ));
                    } else {
                        // TODO: check whether can be casted
                        return ChiAnalysisData::DType(dst_ty.clone());
                    }
                } else {
                    panic!(
                        "Invalid data for cast: {:?} to {:?}",
                        &egraph[*x].data, &egraph[*t].data
                    )
                }
            }
            ChiIR::Constant(_) => ChiAnalysisData::DType(DataType::Float(64)),
            ChiIR::Symbol(s) => {
                if let Ok(_) = s.parse::<i32>() {
                    ChiAnalysisData::DType(DataType::Int(32))
                } else if let Ok(_) = s.parse::<f32>() {
                    ChiAnalysisData::DType(DataType::Float(32))
                } else if let Ok(_) = s.parse::<bool>() {
                    ChiAnalysisData::DType(DataType::Bool)
                } else {
                    ChiAnalysisData::DType(DataType::Unknown)
                }
            }
            ChiIR::Transpose([x]) => {
                if let ChiAnalysisData::DType(dt) = &egraph[*x].data {
                    if let DataType::TensorType(dtype, shape) = dt {
                        if shape.len() == 1 {
                            ChiAnalysisData::DType(dt.clone())
                        } else {
                            ChiAnalysisData::DType(DataType::TensorType(
                                dtype.clone(),
                                vec![shape[1], shape[0]],
                            ))
                        }
                    } else {
                        panic!("Unexpected dtype for transpose: {:?}", dt);
                    }
                } else {
                    panic!("Unexpected data for transpose: {:?}", egraph[*x].data);
                }
            }
            ChiIR::MatMul([x, y]) => {
                if let (
                    ChiAnalysisData::DType(DataType::TensorType(x_dt, x_shape)),
                    ChiAnalysisData::DType(DataType::TensorType(y_dt, y_shape)),
                ) = (&egraph[*x].data, &egraph[*y].data)
                {
                    if x_shape.len() == 2 && y_shape.len() == 2 {
                        if x_shape[1] != y_shape[0] {
                            panic!(
                                "Shape mismatch for matrix multiplication: {:?} {:?}",
                                x_shape, y_shape
                            );
                        }
                        ChiAnalysisData::DType(DataType::TensorType(
                            Box::new(promote_dtype(x_dt, y_dt)),
                            vec![x_shape[0], y_shape[1]],
                        ))
                    } else {
                        panic!("MatMul only supports 2D matrices");
                    }
                } else {
                    panic!(
                        "Unexpected dtype for matrix multiplication: {:?}",
                        (&egraph[*x].data, &egraph[*y].data)
                    );
                }
            }
            ChiIR::While([cond, _]) => {
                if let ChiAnalysisData::DType(DataType::Bool) = &egraph[*cond].data {
                    // TODO: loop variable needed?
                    ChiAnalysisData::LoopVar(HashMap::default())
                } else {
                    panic!(
                        "Unexpected dtype for while loop condition: {:?}",
                        egraph[*cond].data
                    );
                }
            }
            ChiIR::EWAdd([x, y]) => {
                if let (ChiAnalysisData::DType(x_dt), ChiAnalysisData::DType(y_dt)) =
                    (&egraph[*x].data, &egraph[*y].data)
                {
                    if let (
                        DataType::TensorType(x_dtype, x_shape),
                        DataType::TensorType(y_dtype, y_shape),
                    ) = (x_dt, y_dt)
                    {
                        if x_shape != y_shape {
                            panic!("Shape mismatch for element-wise addition");
                        }
                        ChiAnalysisData::DType(promote_dtype(x_dtype, y_dtype))
                    } else {
                        panic!(
                            "Unexpected dtype for element-wise addition: {:?}",
                            (x_dt, y_dt)
                        );
                    }
                } else {
                    panic!(
                        "Unexpected dtype for element-wise addition: {:?}",
                        (&egraph[*x].data, &egraph[*y].data)
                    );
                }
            }
            _ => unimplemented!(),
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
                                   (let y (vector (cast 1.0 f32) (cast 2.0 f32) (cast 3.0 f32)))
                                   (let z (vector (cast 5.0 f32) (cast 6.0 f32) (cast 7.0 f32))))"
            .parse::<RecExpr<ChiIR>>()
            .unwrap();
        println!("{:?}", x);

        println!(
            "{:?}",
            "(== (cast 0 i32) (bitand (cast 1 i32) (cast 2 i32)))"
                .parse::<RecExpr<ChiIR>>()
                .unwrap()
        );
    }

    #[test]
    fn test_add_expr() {
        let x = "(ewadd (vector (cast 1.0 f32) (cast 2.0 f32) (cast 3.0 f32))
                        (vector (cast 5.0 f32) (cast 6.0 f32) (cast 7.0 f32)))"
            .parse::<RecExpr<ChiIR>>()
            .unwrap();
        let y = "(matmul (matrix (vector 1 2 3) (vector 4 5 6) (vector 7 8 9) i32)
                                         (matrix (vector 1 2) (vector 4 5) (vector 7 8) i32)))"
            .parse::<RecExpr<ChiIR>>()
            .unwrap();
        let mut egraph = EGraph::new(ChiAnalysis {
            state: HashMap::default(),
            name_to_shapes: HashMap::default(),
        });
        let id = egraph.add_expr(&x);
        println!("{:?}", egraph[id].data);

        let id = egraph.add_expr(&y);
        println!("{:?}", egraph[id].data);

        let id = egraph.add_expr(
            &"(== (cast 0 i32) (bitand (cast 1 i32) (cast 2 i32)))"
                .parse::<RecExpr<ChiIR>>()
                .unwrap(),
        );
        println!("{:?}", egraph[id].data);
    }
}