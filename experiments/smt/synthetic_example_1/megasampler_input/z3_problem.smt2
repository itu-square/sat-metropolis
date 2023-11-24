; benchmark generated from python API
(set-info :status unknown)
(declare-fun x2 () Int)
(declare-fun x0 () Int)
(declare-fun x1 () Int)
(declare-fun x3 () Int)
(declare-fun x4 () Int)
(assert
 (= x2 30))
(assert
 (<= x0 30))
(assert
 (<= x1 30))
(assert
 (>= x0 0))
(assert
 (>= x1 0))
(assert
 (<= x3 125))
(assert
 (<= x4 125))
(assert
 (>= x3 30))
(assert
 (>= x4 30))
(assert
 (= (+ (+ (+ (+ x0 x1) x2) x3) x4) 190))
(check-sat)
