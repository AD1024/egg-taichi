use crate::language::{ChiAnalysis, ChiIR};
use egg::{rewrite as rw, Rewrite};

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
    #[test]
    fn test_mat_prog() {}
}
