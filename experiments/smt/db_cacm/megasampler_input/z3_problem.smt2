; benchmark generated from python API
(set-info :status unknown)
(declare-fun x0 () Int)
(declare-fun x1 () Int)
(declare-fun x2 () Int)
(declare-fun x3 () Int)
(declare-fun x4 () Int)
(declare-fun x5 () Int)
(declare-fun x6 () Int)
(declare-fun x7 () Int)
(declare-fun x8 () Int)
(declare-fun x9 () Int)
(assert
 (>= x0 0))
(assert
 (<= x0 x1))
(assert
 (<= x1 x2))
(assert
 (= x2 30))
(assert
 (<= x2 x3))
(assert
 (<= x3 x4))
(assert
 (<= x4 125))
(assert
 (= (+ (+ (+ (+ x0 x1) x2) x3) x4) 190))
(assert
 (>= x5 0))
(assert
 (<= x5 1))
(assert
 (>= x6 0))
(assert
 (<= x6 1))
(assert
 (>= x7 0))
(assert
 (<= x7 1))
(assert
 (>= x8 0))
(assert
 (<= x8 1))
(assert
 (>= x9 0))
(assert
 (<= x9 1))
(assert
 (= (+ (+ (+ (+ x5 x6) x7) x8) x9) 3))
(assert
 (let ((?x89 (* x4 x9)))
(let ((?x87 (* x3 x8)))
(let ((?x90 (+ (+ (+ (+ (* x0 x5) (* x1 x6)) (* x2 x7)) ?x87) ?x89)))
(= ?x90 132)))))
(check-sat)
