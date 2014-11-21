(define (make-centroid-predicate name)
    (primitive-eval
        `(define (,name destin-node centroid)
            (EvaluationLink
                (PredicateNode ,(symbol->string name))
                (ListLink
                    (ObjectNode (number->string destin-node))
                    (NumberNode centroid))))))

(define (make-destin-node-predicate name)
    (primitive-eval
        `(define (,name node1 node2)
            (EvaluationLink
                (PredicateNode ,(symbol->string name))
                (ListLink
                    (ObjectNode (number->string node1))
                    (ObjectNode (number->string node2)))))))


;(define-syntax centroid-predicate
;    (syntax-rules ()
;    ((make-destin-predicate name)
;        (define (name destin-node centroid)
;            (EvaluationLink
;                (PredicateNode (symbol->string 'name))
;                (ListLink
;                    (ObjectNode (number->string destin-node))
;                    (NumberNode centroid)))))))

;(define-syntax destin-node-pred
;    (syntax-rules ()
;    ((make-destin-predicate name)
;        (define (name node1 node2)
;            (EvaluationLink
;                (PredicateNode (symbol->string 'name))
;                (ListLink
;                    (ObjectNode (number->string node1))
;                    (ObjectNode (number->string node2))))))))

(define centroid-predicates '(
    hasCentroid
    has0ParentCentroid
    has1ParentCentroid
    has2ParentCentroid
    has3ParentCentroid
    hasNorthNeighborCentroid
    hasSouthNeighborCentroid
    hasEastNeighborCentroid
    hasWestNeighborCentroid
))

(for-each make-centroid-predicate centroid-predicates)

(define destin-node-predicates '(
    has0Parent
    has1Parent
    has2Parent
    has3Parent
    hasNorthNeighbor
    hasSouthNeighbor
    hasEastNeighbor
    hasWestNeighbor
))
(for-each make-destin-node-predicate destin-node-predicates)
