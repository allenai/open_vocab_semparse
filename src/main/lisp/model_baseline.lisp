(define sfe-spec-file (car ARGV))
(define data-name (cadr ARGV))
(define graph-features (create-sfe-feature-computer sfe-spec-file data-name))

(define get-cluster (word word-cluster-dict word-cluster-names cluster-dict clusters)
  (if (dictionary-contains word word-cluster-dict)
    (array-get-ith-element clusters
                           (dictionary-lookup (array-get-ith-element word-cluster-names
                                                                     (dictionary-lookup word word-cluster-dict)) cluster-dict))
    (array-get-ith-element clusters
                           (dictionary-lookup (array-get-ith-element word-cluster-names
                                                                     (dictionary-lookup UNKNOWN-WORD word-cluster-dict)) cluster-dict))))

(define find-related-entities (midsInQuery midRelationsInQuery)
  (array-merge-sets (get-all-related-entities midsInQuery) (find-related-entities-in-graph midRelationsInQuery graph-features)))
