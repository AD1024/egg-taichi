use egg::{self, define_language, Analysis, DidMerge, EGraph, Id};
use ordered_float::NotNan;
use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    ops::Deref,
    str::FromStr,
};

use crate::interp::interp;

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
        "pow" = Pow([Id; 2]),

        "land" = LAnd([Id; 2]),
        "lor" = LOr([Id; 2]),
        "lnot" = LNot([Id; 1]),
        "lxor" = LXor([Id; 2]),

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
        "index" = Index([Id; 2]),
        "matmul" = MatMul([Id; 2]),
        // element-wise addition
        // (ewadd mat/vec mat/vec)
        "ewadd" = EWAdd([Id; 2]),
        "ewmult" = EWMult([Id; 2]),
        // Matrix transpose
        "transpose" = Transpose([Id; 1]),
        "svd" = SVD([Id; 1]),
        "concat" = Concat([Id; 2]),
        "car" = Car([Id; 1]),
        "cdr" = Cdr([Id; 1]),
        "cons" = Cons([Id; 2]),
        "nil" = Nil,

        "seq" = Seq([Id; 2]),

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
        "constant" = Constant([Id; 2]),
        "call" = Call(Vec<Id>),
        TiOperator(TiOperators),
        DataType(DataType),
        Symbol(String),
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum TiOperators {
    Exp,
    Dot,
    Norm,
    NormSqr,
}

impl Display for TiOperators {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TiOperators::Exp => write!(f, "ti.exp"),
            TiOperators::Dot => write!(f, "ti.dot"),
            TiOperators::Norm => write!(f, "ti.norm"),
            TiOperators::NormSqr => write!(f, "ti.norm_sqr"),
        }
    }
}

impl FromStr for TiOperators {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ti.exp" => Ok(TiOperators::Exp),
            "ti.dot" => Ok(TiOperators::Dot),
            "ti.norm" => Ok(TiOperators::Norm),
            "ti.norm_sqr" => Ok(TiOperators::NormSqr),
            _ => Err(format!("Unknown operator: {}", s)),
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum DataType {
    Int(usize),
    Float(usize),
    Bool,
    UInt(usize),
    TensorType(Box<DataType>, Vec<usize>),
    TupleType(Box<DataType>, Box<DataType>),
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
            DataType::TupleType(x, y) => write!(f, "({}, {})", x, y),
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
            x if x.starts_with("(") && x.ends_with(")") => {
                let mut split = x[1..x.len() - 1].split(',');
                let x = split.next().unwrap().trim().parse()?;
                let y = split.next().unwrap().trim().parse()?;
                Ok(DataType::TupleType(Box::new(x), Box::new(y)))
            }
            "unknown" => Ok(DataType::Unknown),
            _ => Err(format!("Unknown data type: {}", s)),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ChiAnalysis {
    pub constants: HashMap<String, ConstData>,
    pub name_to_shapes: HashMap<String, Vec<usize>>,
    pub name_to_type: HashMap<String, DataType>,
}

impl ChiAnalysis {
    pub fn get_dtype(egraph: &EGraph<ChiIR, ChiAnalysis>, id: &Id) -> DataType {
        match &egraph[*id].data.analysis_info {
            AnalysisInfo::DType(dtype) => dtype.clone(),
            _ => panic!("Cannot get dtype: {:?}", egraph[*id]),
        }
    }

    pub fn get_shape(egraph: &EGraph<ChiIR, ChiAnalysis>, id: &Id) -> Vec<usize> {
        match &egraph[*id].data.analysis_info {
            AnalysisInfo::DType(dt) => match dt {
                DataType::TensorType(_, s) => s.clone(),
                _ => panic!("Cannot get shape: {:?}", egraph[*id]),
            },
            AnalysisInfo::Shape(shape) => shape.clone(),
            _ => panic!("Cannot get shape: {:?}", egraph[*id]),
        }
    }

    pub fn check_dtype(egraph: &EGraph<ChiIR, ChiAnalysis>, id: &Id, dtype: &DataType) -> bool {
        let actual = Self::get_dtype(egraph, id);
        actual == *dtype
    }

    pub fn get_constant(egraph: &EGraph<ChiIR, ChiAnalysis>, id: &Id) -> Option<ConstData> {
        egraph[*id].data.consts.clone()
    }
}

#[derive(Debug, Clone)]
pub enum AnalysisInfo {
    DType(DataType),
    Shape(Vec<usize>),
    Binding(Id, Id),
    LoopVar(HashMap<ChiIR, HashSet<String>>),
}

#[derive(Debug, Clone)]
pub enum ConstData {
    Int(i32),
    Float(f32),
    Bool(bool),
    Matrix(Vec<Vec<Box<ConstData>>>),
    Vector(Vec<Box<ConstData>>),
}

impl ConstData {
    pub fn dtype(&self) -> DataType {
        match self {
            ConstData::Int(_) => DataType::Int(32),
            ConstData::Float(_) => DataType::Float(32),
            ConstData::Bool(_) => DataType::Bool,
            _ => unimplemented!()
            // ConstData::Matrix(_) => DataType::TensorType(Box::new(DataType::Float(32)), vec![2, 2]),
            // ConstData::Vector(_) => DataType::TensorType(Box::new(DataType::Float(32)), vec![2]),
        }
    }

    pub fn shl(x: &ConstData, y: &ConstData) -> ConstData {
        match (x, y) {
            (ConstData::Int(x), ConstData::Int(y)) => ConstData::Int(x << y),
            _ => unimplemented!(),
        }
    }

    pub fn shr(x: &ConstData, y: &ConstData) -> ConstData {
        match (x, y) {
            (ConstData::Int(x), ConstData::Int(y)) => ConstData::Int(x >> y),
            _ => unimplemented!(),
        }
    }

    pub fn power(x: &ConstData, y: &ConstData) -> ConstData {
        match (x, y) {
            (ConstData::Int(x), ConstData::Int(y)) => {
                if y < &0 {
                    ConstData::Int(0)
                } else {
                    ConstData::Int(x.pow(*y as u32))
                }
            }
            (ConstData::Float(x), ConstData::Float(y)) => ConstData::Float(x.powf(*y)),
            _ => unimplemented!(),
        }
    }
}

impl core::ops::BitAnd for ConstData {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (ConstData::Bool(x), ConstData::Bool(y)) => ConstData::Bool(x & y),
            (x, y) => panic!("Cannot perform bitand on {:?} and {:?}", x, y),
        }
    }
}

impl core::ops::BitOr for ConstData {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (ConstData::Bool(x), ConstData::Bool(y)) => ConstData::Bool(x | y),
            (x, y) => panic!("Cannot perform bitor on {:?} and {:?}", x, y),
        }
    }
}

impl core::ops::Add for ConstData {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (ConstData::Int(x), ConstData::Int(y)) => ConstData::Int(x + y),
            (ConstData::Float(x), ConstData::Float(y)) => ConstData::Float(x + y),
            (ConstData::Int(x), ConstData::Float(y)) => ConstData::Float(x as f32 + y),
            (ConstData::Float(x), ConstData::Int(y)) => ConstData::Float(x + y as f32),
            (ConstData::Vector(x), ConstData::Vector(y)) => ConstData::Vector(
                x.into_iter()
                    .zip(y)
                    .map(|(x, y)| Box::new((*x.deref()).clone() + (*y.deref()).clone()))
                    .collect(),
            ),
            (ConstData::Matrix(x), ConstData::Matrix(y)) => ConstData::Matrix(
                x.into_iter()
                    .zip(y)
                    .map(|(x, y)| {
                        x.into_iter()
                            .zip(y)
                            .map(|(x, y)| Box::new((*x.deref()).clone() + (*y.deref()).clone()))
                            .collect()
                    })
                    .collect(),
            ),
            (x, y) => panic!("Cannot add {:?} and {:?}", x, y),
        }
    }
}

impl core::ops::Mul for ConstData {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (ConstData::Int(x), ConstData::Int(y)) => ConstData::Int(x * y),
            (ConstData::Float(x), ConstData::Float(y)) => ConstData::Float(x * y),
            (ConstData::Int(x), ConstData::Float(y)) => ConstData::Float(x as f32 * y),
            (ConstData::Float(x), ConstData::Int(y)) => ConstData::Float(x * y as f32),
            (ConstData::Vector(x), ConstData::Vector(y)) => ConstData::Vector(
                x.into_iter()
                    .zip(y)
                    .map(|(x, y)| Box::new((*x.deref()).clone() * (*y.deref()).clone()))
                    .collect(),
            ),
            (ConstData::Matrix(x), ConstData::Matrix(y)) => ConstData::Matrix(
                x.into_iter()
                    .zip(y)
                    .map(|(x, y)| {
                        x.into_iter()
                            .zip(y)
                            .map(|(x, y)| Box::new((*x.deref()).clone() * (*y.deref()).clone()))
                            .collect()
                    })
                    .collect(),
            ),
            (x, y) => panic!("Cannot multiply {:?} and {:?}", x, y),
        }
    }
}

impl core::ops::Sub for ConstData {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (ConstData::Int(x), ConstData::Int(y)) => ConstData::Int(x - y),
            (ConstData::Float(x), ConstData::Float(y)) => ConstData::Float(x - y),
            (ConstData::Int(x), ConstData::Float(y)) => ConstData::Float(x as f32 - y),
            (ConstData::Float(x), ConstData::Int(y)) => ConstData::Float(x - y as f32),
            (ConstData::Vector(x), ConstData::Vector(y)) => ConstData::Vector(
                x.into_iter()
                    .zip(y)
                    .map(|(x, y)| Box::new((*x.deref()).clone() - (*y.deref()).clone()))
                    .collect(),
            ),
            (ConstData::Matrix(x), ConstData::Matrix(y)) => ConstData::Matrix(
                x.into_iter()
                    .zip(y)
                    .map(|(x, y)| {
                        x.into_iter()
                            .zip(y)
                            .map(|(x, y)| Box::new((*x.deref()).clone() - (*y.deref()).clone()))
                            .collect()
                    })
                    .collect(),
            ),
            (x, y) => panic!("Cannot subtract {:?} and {:?}", x, y),
        }
    }
}

impl core::ops::Div for ConstData {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (ConstData::Int(x), ConstData::Int(y)) => ConstData::Int(x / y),
            (ConstData::Float(x), ConstData::Float(y)) => ConstData::Float(x / y),
            (ConstData::Int(x), ConstData::Float(y)) => ConstData::Float(x as f32 / y),
            (ConstData::Float(x), ConstData::Int(y)) => ConstData::Float(x / y as f32),
            (ConstData::Vector(x), ConstData::Vector(y)) => ConstData::Vector(
                x.into_iter()
                    .zip(y)
                    .map(|(x, y)| Box::new((*x.deref()).clone() / (*y.deref()).clone()))
                    .collect(),
            ),
            (ConstData::Matrix(x), ConstData::Matrix(y)) => ConstData::Matrix(
                x.into_iter()
                    .zip(y)
                    .map(|(x, y)| {
                        x.into_iter()
                            .zip(y)
                            .map(|(x, y)| Box::new((*x.deref()).clone() / (*y.deref()).clone()))
                            .collect()
                    })
                    .collect(),
            ),
            (x, y) => panic!("Cannot divide {:?} and {:?}", x, y),
        }
    }
}

impl Into<ConstData> for i32 {
    fn into(self) -> ConstData {
        ConstData::Int(self)
    }
}

impl Into<ConstData> for f32 {
    fn into(self) -> ConstData {
        ConstData::Float(self)
    }
}

impl Into<ConstData> for bool {
    fn into(self) -> ConstData {
        ConstData::Bool(self)
    }
}

#[derive(Debug, Clone)]
pub struct ChiAnalysisData {
    pub analysis_info: AnalysisInfo,
    pub consts: Option<ConstData>,
}

fn promote_dtype(x: &DataType, y: &DataType) -> DataType {
    match x {
        DataType::TupleType(car, cdr) => match y {
            DataType::TupleType(car2, cdr2) => DataType::TupleType(
                Box::new(promote_dtype(car, car2)),
                Box::new(promote_dtype(cdr, cdr2)),
            ),
            _ => panic!("Cannot promote tuple type with non-tuple type"),
        },
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
            DataType::TupleType(_, _) => panic!("Cannot promote tuple type with int"),
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
            DataType::TupleType(_, _) => panic!("Cannot promote tuple type with UInt"),
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
            DataType::TupleType(_, _) => panic!("Cannot promote tuple type with float"),
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
        let merge_const = if lhs.consts.is_some() || rhs.consts.is_some() {
            if lhs.consts.is_none() {
                lhs.consts = rhs.consts.clone();
                DidMerge(true, false)
            } else {
                DidMerge(false, rhs.consts.is_none())
            }
        } else {
            DidMerge(false, false)
        };
        merge_const
            | match (&lhs.analysis_info, &rhs.analysis_info) {
                (AnalysisInfo::DType(l), AnalysisInfo::DType(r)) => {
                    if l == r {
                        DidMerge(false, false)
                    } else {
                        lhs.analysis_info = AnalysisInfo::DType(promote_dtype(l, r));
                        DidMerge(true, false)
                    }
                }
                (AnalysisInfo::Shape(l), AnalysisInfo::Shape(r)) => {
                    if l == r {
                        DidMerge(false, false)
                    } else {
                        panic!("Shape mismatch: {:?} vs {:?}", l, r);
                    }
                }
                (AnalysisInfo::LoopVar(l), AnalysisInfo::LoopVar(r)) => {
                    let mut modified = false;
                    let mut new_vars = l.clone();
                    for (k, v) in r {
                        if l.contains_key(&k) {
                            let loop_vars = l.get(&k).unwrap();
                            for var in v {
                                if !loop_vars.contains(var) {
                                    panic!("Loop variable mismatch: {:?} vs {:?}", loop_vars, v);
                                }
                            }
                        } else {
                            new_vars.insert(k.clone(), v.clone());
                            modified = true;
                        }
                    }
                    lhs.analysis_info = AnalysisInfo::LoopVar(new_vars);
                    if !modified {
                        DidMerge(false, false)
                    } else {
                        DidMerge(true, true)
                    }
                }
                (lhs, rhs) => panic!("Cannot merge {:?} and {:?}", lhs, rhs),
            }
    }

    fn make(egraph: &mut egg::EGraph<ChiIR, Self>, enode: &ChiIR) -> ChiAnalysisData {
        match enode {
            ChiIR::Call(params) => {
                assert!(params.len() >= 2);
                let op = params.first().unwrap();
                assert_eq!(egraph[*op].nodes.len(), 1);
                let op = &egraph[*op].nodes[0];
                let args = &params[1..]
                    .iter()
                    .map(|x| &egraph[*x].data)
                    .collect::<Vec<_>>()[..];
                match op {
                    ChiIR::TiOperator(TiOperators::Exp) => match args {
                        [val] => interp(TiOperators::Exp, vec![val]),
                        _ => panic!("ti.exp only takes 1 argument"),
                    },
                    ChiIR::TiOperator(TiOperators::Dot) => match args {
                        [x, y] => interp(TiOperators::Dot, vec![x, y]),
                        _ => panic!("ti.dot only takes 2 arguments"),
                    },
                    ChiIR::TiOperator(TiOperators::Norm) => match args {
                        [x] => interp(TiOperators::Norm, vec![x]),
                        _ => panic!("ti.norm only takes 1 argument"),
                    },
                    ChiIR::TiOperator(TiOperators::NormSqr) => match args {
                        [x] => interp(TiOperators::NormSqr, vec![x]),
                        _ => panic!("ti.norm_sqr only takes 1 argument"),
                    },
                    _ => panic!("Unrecognized ti operator: {:?}", op),
                }
            }
            ChiIR::SMinus([x, y])
            | ChiIR::SMult([x, y])
            | ChiIR::SDiv([x, y])
            | ChiIR::SMod([x, y])
            | ChiIR::SAdd([x, y])
            | ChiIR::Pow([x, y]) => match (
                &egraph[*x].data.analysis_info,
                &egraph[*y].data.analysis_info,
            ) {
                (AnalysisInfo::DType(x_dtype), AnalysisInfo::DType(y_dtype)) => {
                    if let DataType::TensorType(dt, _) = x_dtype {
                        return ChiAnalysisData {
                            analysis_info: AnalysisInfo::DType(promote_dtype(dt, y_dtype)),
                            consts: None,
                        };
                    } else if let DataType::TensorType(dt, _) = y_dtype {
                        return ChiAnalysisData {
                            analysis_info: AnalysisInfo::DType(promote_dtype(x_dtype, dt)),
                            consts: None,
                        };
                    }
                    let dtype = promote_dtype(x_dtype, y_dtype);
                    ChiAnalysisData {
                        analysis_info: AnalysisInfo::DType(dtype),
                        consts: None,
                    }
                }
                _ => panic!("Unexpected dtype"),
            },
            ChiIR::Gt([_x, _y])
            | ChiIR::Lt([_x, _y])
            | ChiIR::Equals([_x, _y])
            | ChiIR::Gte([_x, _y])
            | ChiIR::Lte([_x, _y]) => ChiAnalysisData {
                analysis_info: AnalysisInfo::DType(DataType::Bool),
                consts: None,
            },
            ChiIR::LAnd([x, y]) | ChiIR::LOr([x, y]) | ChiIR::LXor([x, y]) => {
                let x_dtype = ChiAnalysis::get_dtype(egraph, x);
                let y_dtype = ChiAnalysis::get_dtype(egraph, y);
                assert_eq!(x_dtype, DataType::Bool);
                assert_eq!(y_dtype, DataType::Bool);
                ChiAnalysisData {
                    analysis_info: AnalysisInfo::DType(DataType::Bool),
                    consts: None,
                }
            }
            ChiIR::LNot([x]) => {
                let x_dtype = ChiAnalysis::get_dtype(egraph, x);
                assert_eq!(x_dtype, DataType::Bool);
                ChiAnalysisData {
                    analysis_info: AnalysisInfo::DType(DataType::Bool),
                    consts: None,
                }
            }
            ChiIR::BitAnd([x, y])
            | ChiIR::BitOr([x, y])
            | ChiIR::BitXor([x, y])
            | ChiIR::BitShl([x, y])
            | ChiIR::BitShr([x, y]) => match (
                &egraph[*x].data.analysis_info,
                &egraph[*y].data.analysis_info,
            ) {
                (AnalysisInfo::DType(x_dtype), AnalysisInfo::DType(y_dtype)) => {
                    let dtype = promote_dtype(&x_dtype, &y_dtype);
                    if let DataType::Int(_) = dtype {
                        ChiAnalysisData {
                            analysis_info: AnalysisInfo::DType(dtype),
                            consts: None,
                        }
                    } else {
                        panic!("Unexpected dtype: {:?}", dtype);
                    }
                }
                _ => panic!(
                    "Unexpected data for bit operation: {:?} vs {:?}",
                    &egraph[*x].data, &egraph[*y].data
                ),
            },
            ChiIR::BitNot([x]) => match &egraph[*x].data.analysis_info {
                AnalysisInfo::DType(x_dtype) => {
                    if let DataType::Int(_) = x_dtype {
                        ChiAnalysisData {
                            analysis_info: AnalysisInfo::DType(x_dtype.clone()),
                            consts: None,
                        }
                    } else {
                        panic!("Unexpected dtype: {:?}", x_dtype);
                    }
                }
                _ => panic!("Unexpected data for bit operation: {:?}", &egraph[*x].data),
            },
            ChiIR::Matrix(m) => {
                assert!(m.len() > 1);
                let dt = &egraph[*m.last().unwrap()].data.analysis_info;
                let decl_type = if let AnalysisInfo::DType(dt) = dt {
                    dt.clone()
                } else {
                    DataType::Unknown
                };
                let dtype = &egraph[m[0]].data;
                if let AnalysisInfo::DType(dt) = &dtype.analysis_info {
                    if let DataType::TensorType(dtype, v_shape) = dt {
                        let mut row_cnt = 0;
                        let mut final_dtype = dtype.deref().clone();
                        m.iter()
                            .take(m.len())
                            .for_each(|x| match &egraph[*x].data.analysis_info {
                                AnalysisInfo::DType(dt) => {
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
                        return ChiAnalysisData {
                            analysis_info: AnalysisInfo::DType(DataType::TensorType(
                                Box::new(if decl_type == DataType::Unknown {
                                    final_dtype
                                } else {
                                    decl_type
                                }),
                                vec![row_cnt, v_shape[0]],
                            )),
                            consts: None,
                        };
                    }
                }
                panic!("DType error for constructing matrix: {:?}", dtype);
            }
            ChiIR::Vector(v) => {
                assert!(v.len() > 1);
                let dt = &egraph[*v.last().unwrap()].data.analysis_info;
                let mut decl_dtype = if let AnalysisInfo::DType(dt) = dt {
                    dt.clone()
                } else {
                    DataType::Unknown
                };
                let mut elem_count = 0;
                v.iter().take(v.len()).for_each(|x| {
                    if let AnalysisInfo::DType(dt) = &egraph[*x].data.analysis_info {
                        decl_dtype = promote_dtype(&decl_dtype, dt);
                        elem_count += 1;
                    }
                });
                // ChiAnalysisData::DType(DataType::TensorType(Box::new(decl_dtype), vec![elem_count]))
                ChiAnalysisData {
                    analysis_info: AnalysisInfo::DType(DataType::TensorType(
                        Box::new(decl_dtype),
                        vec![elem_count],
                    )),
                    consts: None,
                }
            }
            ChiIR::DataType(dtype) => ChiAnalysisData {
                analysis_info: AnalysisInfo::DType(dtype.clone()),
                consts: None,
            },
            ChiIR::Cast([x, t]) => {
                if let (AnalysisInfo::DType(x_dt), AnalysisInfo::DType(dst_ty)) = (
                    &egraph[*x].data.analysis_info,
                    &egraph[*t].data.analysis_info,
                ) {
                    if let DataType::TensorType(_x_elem_type, shape) = x_dt {
                        // TODO: check whether can be cast
                        // return ChiAnalysisData::DType(DataType::TensorType(
                        //     Box::new(dst_ty.clone()),
                        //     shape.clone(),
                        // ));
                        return ChiAnalysisData {
                            analysis_info: AnalysisInfo::DType(DataType::TensorType(
                                Box::new(dst_ty.clone()),
                                shape.clone(),
                            )),
                            consts: None,
                        };
                    } else {
                        // TODO: check whether can be casted
                        // return ChiAnalysisData::DType(dst_ty.clone());
                        return ChiAnalysisData {
                            analysis_info: AnalysisInfo::DType(dst_ty.clone()),
                            consts: None,
                        };
                    }
                } else {
                    panic!(
                        "Invalid data for cast: {:?} to {:?}",
                        &egraph[*x].data, &egraph[*t].data
                    )
                }
            }
            ChiIR::Constant([v, t]) => ChiAnalysisData {
                analysis_info: egraph[*t].data.analysis_info.clone(),
                consts: Some(egraph[*v].data.consts.clone().unwrap()),
            },
            ChiIR::Symbol(s) => {
                if let Ok(x) = s.parse::<i32>() {
                    ChiAnalysisData {
                        analysis_info: AnalysisInfo::DType(DataType::Int(32)),
                        consts: Some(x.into()),
                    }
                } else if let Ok(x) = s.parse::<f32>() {
                    ChiAnalysisData {
                        analysis_info: AnalysisInfo::DType(DataType::Float(32)),
                        consts: Some(x.into()),
                    }
                } else if let Ok(x) = s.parse::<bool>() {
                    // ChiAnalysisData::DType(DataType::Bool)
                    ChiAnalysisData {
                        analysis_info: AnalysisInfo::DType(DataType::Bool),
                        consts: Some(x.into()),
                    }
                } else if let Some(constant) = egraph.analysis.constants.get(s) {
                    ChiAnalysisData {
                        analysis_info: AnalysisInfo::DType(constant.dtype()),
                        consts: Some(constant.clone()),
                    }
                } else if let Some(dt) = egraph.analysis.name_to_type.get(s) {
                    if let Some(shape) = egraph.analysis.name_to_shapes.get(s) {
                        ChiAnalysisData {
                            analysis_info: AnalysisInfo::DType(DataType::TensorType(
                                Box::new(dt.clone()),
                                shape.clone(),
                            )),
                            consts: None,
                        }
                    } else {
                        ChiAnalysisData {
                            analysis_info: AnalysisInfo::DType(dt.clone()),
                            consts: None,
                        }
                    }
                } else {
                    ChiAnalysisData {
                        analysis_info: AnalysisInfo::DType(DataType::Unknown),
                        consts: None,
                    }
                }
            }
            ChiIR::Transpose([x]) => {
                if let AnalysisInfo::DType(dt) = &egraph[*x].data.analysis_info {
                    if let DataType::TensorType(dtype, shape) = dt {
                        if shape.len() == 1 {
                            ChiAnalysisData {
                                analysis_info: AnalysisInfo::DType(dt.clone()),
                                consts: None,
                            }
                        } else {
                            ChiAnalysisData {
                                analysis_info: AnalysisInfo::DType(DataType::TensorType(
                                    dtype.clone(),
                                    vec![shape[1], shape[0]],
                                )),
                                consts: None,
                            }
                        }
                    } else {
                        panic!("Unexpected dtype for transpose: {:?}", dt);
                    }
                } else {
                    panic!("Unexpected data for transpose: {:?}", egraph[*x].data);
                }
            }
            ChiIR::SVD([x]) => {
                if let AnalysisInfo::DType(dt) = &egraph[*x].data.analysis_info {
                    if let DataType::TensorType(dtype, shape) = dt {
                        let (m, n) = (shape[0], shape[1]);
                        assert_eq!(m, n);
                        let ret_mat_dtype = DataType::TensorType(dtype.clone(), vec![n, n]);
                        ChiAnalysisData {
                            analysis_info: AnalysisInfo::DType(DataType::TupleType(
                                Box::new(DataType::TupleType(
                                    Box::new(ret_mat_dtype.clone()),
                                    Box::new(ret_mat_dtype.clone()),
                                )),
                                Box::new(ret_mat_dtype),
                            )),
                            consts: None,
                        }
                    } else {
                        panic!("Unexpected dtype for SVD: {:?}", dt);
                    }
                } else {
                    panic!("Unexpected data for SVD: {:?}", egraph[*x].data);
                }
            }
            ChiIR::MatMul([x, y]) => {
                if let (
                    AnalysisInfo::DType(DataType::TensorType(x_dt, x_shape)),
                    AnalysisInfo::DType(DataType::TensorType(y_dt, y_shape)),
                ) = (
                    &egraph[*x].data.analysis_info,
                    &egraph[*y].data.analysis_info,
                ) {
                    if x_shape.len() == 2 && y_shape.len() == 2 {
                        if x_shape[1] != y_shape[0] {
                            panic!(
                                "Shape mismatch for matrix multiplication: {:?} {:?}",
                                x_shape, y_shape
                            );
                        }
                        ChiAnalysisData {
                            analysis_info: AnalysisInfo::DType(DataType::TensorType(
                                Box::new(promote_dtype(x_dt, y_dt)),
                                vec![x_shape[0], y_shape[1]],
                            )),
                            consts: None,
                        }
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
            ChiIR::Car([x]) => {
                let x_dtype = ChiAnalysis::get_dtype(egraph, x);
                if let DataType::TupleType(x, _) = x_dtype {
                    return ChiAnalysisData {
                        analysis_info: AnalysisInfo::DType(*x.clone()),
                        consts: None,
                    };
                } else {
                    panic!("Unexpected data for car: {:?}", x_dtype);
                }
            }
            ChiIR::Cdr([x]) => {
                let x_dtype = ChiAnalysis::get_dtype(egraph, x);
                if let DataType::TupleType(_, x) = x_dtype {
                    return ChiAnalysisData {
                        analysis_info: AnalysisInfo::DType(*x.clone()),
                        consts: None,
                    };
                } else {
                    panic!("Unexpected data for cdr: {:?}", x_dtype);
                }
            }
            ChiIR::Cons([x, y]) => {
                let x_dtype = ChiAnalysis::get_dtype(egraph, x);
                let y_dtype = ChiAnalysis::get_dtype(egraph, y);
                ChiAnalysisData {
                    analysis_info: AnalysisInfo::DType(DataType::TupleType(
                        Box::new(x_dtype),
                        Box::new(y_dtype),
                    )),
                    consts: None,
                }
            }
            ChiIR::Seq([_, _]) => {
                // TODO: Maybe useful when propagating static analysis info
                ChiAnalysisData {
                    analysis_info: AnalysisInfo::DType(DataType::Unknown),
                    consts: None,
                }
            }
            ChiIR::Load([v]) => {
                if let AnalysisInfo::DType(dt) = &egraph[*v].data.analysis_info {
                    ChiAnalysisData {
                        analysis_info: AnalysisInfo::DType(dt.clone()),
                        consts: None,
                    }
                } else {
                    panic!("Unexpected data for load: {:?}", egraph[*v].data);
                }
            }
            ChiIR::Store([_, val]) => {
                if let AnalysisInfo::DType(dt) = &egraph[*val].data.analysis_info {
                    ChiAnalysisData {
                        analysis_info: AnalysisInfo::DType(dt.clone()),
                        consts: None,
                    }
                } else {
                    panic!("Unexpected data for store: {:?}", egraph[*val].data);
                }
            }
            ChiIR::While([cond, _]) => {
                if let AnalysisInfo::DType(DataType::Bool) = &egraph[*cond].data.analysis_info {
                    // TODO: loop variable needed?
                    ChiAnalysisData {
                        analysis_info: AnalysisInfo::LoopVar(HashMap::default()),
                        consts: None,
                    }
                } else {
                    panic!(
                        "Unexpected dtype for while loop condition: {:?}",
                        egraph[*cond].data
                    );
                }
            }
            ChiIR::Index([x, i]) => {
                if let AnalysisInfo::DType(DataType::TensorType(x_dtype, x_shape)) =
                    &egraph[*x].data.analysis_info
                {
                    let dtype = if x_shape.len() == 1 {
                        x_dtype.deref().clone()
                    } else {
                        DataType::TensorType(
                            Box::new(x_dtype.deref().clone()),
                            x_shape[1..].to_vec(),
                        )
                    };
                    match &egraph[*i].data.analysis_info {
                        AnalysisInfo::DType(DataType::Int(_)) => {
                            if let Some(idx) = &egraph[*i].data.consts {
                                if let ConstData::Int(idx) = idx {
                                    if *idx < 0 {
                                        panic!("Negative index: {:?}", idx);
                                    }
                                    if *idx >= x_shape.len() as i32 {
                                        panic!("Index out of bounds: {:?}", idx);
                                    }
                                    let consts = match ChiAnalysis::get_constant(egraph, x) {
                                        Some(ConstData::Matrix(x)) => {
                                            Some(ConstData::Vector(x[*idx as usize].clone()))
                                        }
                                        Some(ConstData::Vector(x)) => {
                                            Some((*x[*idx as usize].deref()).clone())
                                        }
                                        _ => None,
                                    };
                                    ChiAnalysisData {
                                        analysis_info: AnalysisInfo::DType(dtype),
                                        consts,
                                    }
                                } else {
                                    panic!("Non-integer index: {:?}", idx);
                                }
                            } else {
                                ChiAnalysisData {
                                    analysis_info: AnalysisInfo::DType(dtype),
                                    consts: None,
                                }
                            }
                        }
                        others => panic!("Non-integer index: {:?}", others),
                    }
                } else {
                    panic!("Unexpected dtype for indexing: {:?}", egraph[*x].data);
                }
            }
            ChiIR::EWAdd([x, y]) | ChiIR::EWMult([x, y]) => {
                if let (AnalysisInfo::DType(x_dt), AnalysisInfo::DType(y_dt)) = (
                    &egraph[*x].data.analysis_info,
                    &egraph[*y].data.analysis_info,
                ) {
                    if let (
                        DataType::TensorType(x_dtype, x_shape),
                        DataType::TensorType(y_dtype, y_shape),
                    ) = (x_dt, y_dt)
                    {
                        if x_shape != y_shape {
                            panic!("Shape mismatch for element-wise addition");
                        }
                        // ChiAnalysisData::DType(promote_dtype(x_dtype, y_dtype))
                        ChiAnalysisData {
                            analysis_info: AnalysisInfo::DType(DataType::TensorType(
                                Box::new(promote_dtype(x_dtype, y_dtype)),
                                x_shape.clone(),
                            )),
                            consts: None,
                        }
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
            ChiIR::IfThenElse([cond, b1, b2]) => {
                assert!(ChiAnalysis::check_dtype(egraph, cond, &DataType::Bool));
                let ret_type = ChiAnalysis::get_dtype(egraph, b2);
                assert!(ChiAnalysis::check_dtype(egraph, b1, &ret_type));
                ChiAnalysisData {
                    analysis_info: AnalysisInfo::DType(ret_type),
                    consts: None,
                }
            }
            ChiIR::Let([x, v]) => ChiAnalysisData {
                analysis_info: AnalysisInfo::Binding(x.clone(), v.clone()),
                consts: egraph[*v].data.consts.clone(),
            },
            ChiIR::Compute(xs) => {
                assert!(xs.len() > 0);
                let e = xs[0];
                let enode = &egraph[e].nodes[0];
                let mut bindings = HashMap::new();
                for x in xs.iter().skip(1) {
                    if let AnalysisInfo::Binding(x, v) = &egraph[*x].data.analysis_info {
                        if let AnalysisInfo::DType(dt) = &egraph[*v].data.analysis_info {
                            for sym in &egraph[*x].nodes {
                                if let ChiIR::Symbol(s) = sym {
                                    bindings.insert(s.clone(), dt.clone());
                                } else {
                                    panic!("Unknown binding symbol: {:?}", sym);
                                }
                            }
                        }
                    } else {
                        panic!("Unexpected data for compute: {:?}", egraph[*x].data);
                    }
                }
                let analysis = ChiAnalysis {
                    constants: egraph.analysis.constants.clone(),
                    name_to_shapes: HashMap::new(),
                    name_to_type: bindings,
                };
                let mut egraph = egraph.clone();
                egraph.analysis = analysis;
                ChiAnalysis::make(&mut egraph, &enode)
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
        // let _x = "(compute (matmul x (matmul x y)) (let x (matrix ...)))";
        let y = "(matmul (matrix (vector 1 2 3) (vector 4 5 6) (vector 7 8 9) i32)
                                         (matrix (vector 1 2) (vector 4 5) (vector 7 8) i32)))"
            .parse::<RecExpr<ChiIR>>()
            .unwrap();
        let mut egraph = EGraph::new(ChiAnalysis {
            constants: HashMap::default(),
            name_to_shapes: HashMap::default(),
            name_to_type: HashMap::default(),
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

    #[test]
    fn test_binding() {
        let expr: RecExpr<ChiIR> = "(let x (matrix (vector 1 2 3) (vector 3 4 5) f32) 
                                        (let y (matrix (vector 1 2) (vector 2 3) (vector 3 4) f32)
                                            (Call ti.dot x y)))"
            .parse()
            .unwrap();
    }
}
