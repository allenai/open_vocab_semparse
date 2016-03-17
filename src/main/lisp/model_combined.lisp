(define sfe-spec-file (car ARGV))
(define data-name (cadr ARGV))
(define graph-features (create-sfe-feature-computer sfe-spec-file data-name))

(define word-family (word-parameters entity-parameters word-graph-parameters)
  (lambda (word)
    (lambda (entity)
      (if (dictionary-contains entity entities)
        (let ((var (make-entity-var entity))
              (word_or_unknown (if (dictionary-contains word cat-words) word UNKNOWN-WORD)))
          (make-inner-product-classifier
            var #t (get-cat-word-params word word-parameters) (get-entity-params entity entity-parameters))
          (make-featurized-classifier
            var (get-entity-features entity word_or_unknown graph-features) (get-cat-word-params word_or_unknown word-graph-parameters))
          var)
        ; The bilinear model needs to have a vector learned for each entity, but the graph features
        ; model doesn't.  So we just score the graph features model when the entity is not in our
        ; dictionary.
        (let ((var (make-entity-var entity))
              (word_or_unknown (if (dictionary-contains word cat-words) word UNKNOWN-WORD)))
          (make-featurized-classifier
            var (get-entity-features entity word_or_unknown graph-features) (get-cat-word-params word_or_unknown word-graph-parameters))
          var))
      )))

(define word-rel-family (word-rel-params entity-tuple-params word-rel-graph-parameters)
  (define word-rel (word)
    (lambda (entity1 entity2)
      (if (dictionary-contains (list entity1 entity2) entity-tuples)
        (let ((var (make-entity-var (cons entity1 entity2)))
              (word_or_unknown (if (dictionary-contains word rel-words) word UNKNOWN-WORD)))
          (make-inner-product-classifier
            var #t (get-rel-word-params word word-rel-params)
            (get-entity-tuple-params entity1 entity2 entity-tuple-params))
          (make-featurized-classifier
            var (get-entity-tuple-features entity1 entity2 word_or_unknown graph-features) (get-rel-word-params word_or_unknown word-rel-graph-parameters))
          var)
        ; The bilinear model needs to have a vector learned for each entity pair, but the graph
        ; features model doesn't.  So we just score the graph features model when the entity pair
        ; is not in our dictionary.
        (let ((var (make-entity-var (cons entity1 entity2)))
              (word_or_unknown (if (dictionary-contains word rel-words) word UNKNOWN-WORD)))
          (make-featurized-classifier
            var (get-entity-tuple-features entity1 entity2 word_or_unknown graph-features) (get-rel-word-params word_or_unknown word-rel-graph-parameters))
          var)
        )
      ))
  word-rel)

(define get-word-cat (parameters)
  (let ((word-parameters (get-ith-parameter parameters 0))
        (word-graph-parameters (get-ith-parameter parameters 1))
        (entity-parameters (get-ith-parameter parameters 2))
        (word-cat (word-family word-parameters entity-parameters word-graph-parameters)))
    word-cat))

(define get-word-rel (parameters)
  (let ((word-rel-parameters (get-ith-parameter parameters 3))
        (word-rel-graph-parameters (get-ith-parameter parameters 4))
        (entity-tuple-parameters (get-ith-parameter parameters 5))
        (word-rel (word-rel-family word-rel-parameters entity-tuple-parameters word-rel-graph-parameters)))
    word-rel))

(define expression-family (parameters)
  (let ((word-cat (get-word-cat parameters))
        (word-rel (get-word-rel parameters)))
    (define expression-evaluator (expression entities)
      (eval expression))
    expression-evaluator))

(define word-ranking-family (word-parameters entity-parameters word-graph-parameters)
  (lambda (word)
    (lambda (entity neg-entity)
      (if (dictionary-contains entity entities)
        (let ((var (make-entity-var entity))
              (word_or_unknown (if (dictionary-contains word cat-words) word UNKNOWN-WORD)))
          (make-ranking-inner-product-classifier
            var #t (get-cat-word-params word word-parameters) (get-entity-params entity entity-parameters)
            (get-entity-params neg-entity entity-parameters))
          (make-featurized-classifier
            var (get-entity-feature-difference entity neg-entity word_or_unknown graph-features) (get-cat-word-params word_or_unknown word-graph-parameters))
          var)
        (let ((var (make-entity-var entity))
              (word_or_unknown (if (dictionary-contains word cat-words) word UNKNOWN-WORD)))
          (make-featurized-classifier
            var (get-entity-feature-difference entity neg-entity word_or_unknown graph-features) (get-cat-word-params word_or_unknown word-graph-parameters))
          var))
      )))

(define word-rel-ranking-family (word-rel-params entity-tuple-params word-rel-graph-parameters)
  (define word-rel (word)
    (lambda (entity1 neg-entity1 entity2 neg-entity2)
      (if (dictionary-contains (list entity1 entity2) entity-tuples)
        (let ((var (make-entity-var (cons entity1 entity2)))
              (word_or_unknown (if (dictionary-contains word rel-words) word UNKNOWN-WORD)))
          (make-ranking-inner-product-classifier
            var #t (get-rel-word-params word word-rel-params)
            (get-entity-tuple-params entity1 entity2 entity-tuple-params)
            (get-entity-tuple-params neg-entity1 neg-entity2 entity-tuple-params))
          (make-featurized-classifier
            var (get-entity-tuple-feature-difference entity1 entity2 neg-entity1 neg-entity2 word_or_unknown graph-features) (get-rel-word-params word_or_unknown word-rel-graph-parameters))
          var)
        (let ((var (make-entity-var (cons entity1 entity2)))
              (word_or_unknown (if (dictionary-contains word rel-words) word UNKNOWN-WORD)))
          (make-featurized-classifier
            var (get-entity-tuple-feature-difference entity1 entity2 neg-entity1 neg-entity2 word_or_unknown graph-features) (get-rel-word-params word_or_unknown word-rel-graph-parameters))
          var))
      ))
  word-rel)

(define get-ranking-word-cat (parameters)
  (let ((word-parameters (get-ith-parameter parameters 0))
        (word-graph-parameters (get-ith-parameter parameters 1))
        (entity-parameters (get-ith-parameter parameters 2))
        (word-cat (word-ranking-family word-parameters entity-parameters word-graph-parameters)))
    word-cat))

(define get-ranking-word-rel (parameters)
  (let ((word-rel-parameters (get-ith-parameter parameters 3))
        (word-rel-graph-parameters (get-ith-parameter parameters 4))
        (entity-tuple-parameters (get-ith-parameter parameters 5))
        (word-rel (word-rel-ranking-family word-rel-parameters entity-tuple-parameters word-rel-graph-parameters)))
    word-rel))

(define expression-ranking-family (parameters)
  (let ((word-cat (get-ranking-word-cat parameters))
        (word-rel (get-ranking-word-rel parameters))
        (expression-evaluator (get-expression-evaluator word-cat word-rel)))
    expression-evaluator))


(define expression-parameters
  (make-parameter-list (list (make-parameter-list (array-map (lambda (x) (make-vector-parameters latent-dimensionality))
                                                             (dictionary-to-array cat-words)))
                             (make-parameter-list (array-map (lambda (x) (make-featurized-classifier-parameters (list true-false) (get-cat-word-feature-list x graph-features)))
                                                             (dictionary-to-array cat-words)))
                             (make-parameter-list (array-map (lambda (x) (make-vector-parameters latent-dimensionality))
                                                             (dictionary-to-array entities)))
                             (make-parameter-list (array-map (lambda (x) (make-vector-parameters latent-dimensionality))
                                                             (dictionary-to-array rel-words)))
                             (make-parameter-list (array-map (lambda (x) (make-featurized-classifier-parameters (list true-false) (get-rel-word-feature-list x graph-features)))
                                                             (dictionary-to-array rel-words)))
                             (make-parameter-list (array-map (lambda (x) (make-vector-parameters latent-dimensionality))
                                                             (dictionary-to-array entity-tuples)))
                             )))

(define find-related-entities (midsInQuery midRelationsInQuery)
  (array-merge-sets (get-all-related-entities midsInQuery) (find-related-entities-in-graph midRelationsInQuery graph-features)))
