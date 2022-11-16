use crate::language::{ChiAnalysis, ChiIR};
use egg::{rewrite as rw, Rewrite};

pub fn alg_simp() -> Vec<Rewrite<ChiIR, ChiAnalysis>> {
    vec![
        rw!("smult-id"; "(smult 1 ?x)" => "?x"),
        rw!("smult-0"; "(smult 0 ?x)" => "0"),
        rw!("smult-comm"; "(smul ?x ?y)" => "(smul ?y ?x)"),
        rw!("smult-assoc"; "(smul (smul ?x ?y) ?z)" => "(smul ?x (smul ?y ?z))"),
        rw!("div-1"; "(div ?x 1)" => "?x"),
        rw!("div-cast-1"; "(div ?x (cast 1 ?t))" => "?x"),
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
