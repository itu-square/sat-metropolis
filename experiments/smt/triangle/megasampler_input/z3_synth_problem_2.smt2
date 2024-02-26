; benchmark generated from python API
(set-info :status unknown)
(declare-fun x0 () Int)
(declare-fun x1 () Int)
(declare-fun x2 () Int)
(assert
 (>= x0 0))
(assert
 (>= x1 0))
(assert
 (>= x2 0))
(assert
 (<= x0 3))
(assert
 (<= x1 3))
(assert
 (<= x2 6))
(assert
 (= (+ x0 x1) x2))
(check-sat)
