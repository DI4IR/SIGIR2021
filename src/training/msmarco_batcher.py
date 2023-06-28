class MSMarcoBatcher():
    def __init__(self, args):
        self.triples = self._load_triples(args.triples)
        self.queries = self._load_queries(args.queries)
        self.collection = self._load_collection(args.collection)

    def _load_triples(self, path):
        triples = []
        with open(path) as f:
            for line_idx, line in enumerate(f):
                qid, pos, neg = line.strip().split("\t")
                triples.append((qid, pos, neg))
        return triples

    def _load_queries(self, path):
        queries = {}
        with open(path) as f:
            for line in f:
                qid, query = line.strip().split('\t')
                qid = int(qid)
                queries[qid] = query
        return queries

    def _load_collection(self, path):
        collection = []
        with open(path) as f:
            for line_idx, line in enumerate(f):
                pid, passage, = line.strip().split('\t')
                assert pid == 'id' or int(pid) == line_idx
                collection.append(passage)

        return collection

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        query, pos, neg = self.triples[idx]
        qid, pos_id, neg_id = int(query), int(pos), int(neg)
        query, pos, neg = self.queries[int(query)], self.collection[int(
            pos)], self.collection[int(neg)]
        return query, pos, neg, qid, pos_id, neg_id
