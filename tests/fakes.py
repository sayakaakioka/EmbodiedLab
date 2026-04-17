from copy import deepcopy


class FakeSnapshot:
	def __init__(self, data):
		self._data = data
		self.exists = data is not None

	def to_dict(self):
		return self._data


class FakeDocument:
	def __init__(self, store: dict, document_id: str):
		self.store = store
		self.document_id = document_id
		self.payloads = []

	def get(self):
		return FakeSnapshot(self.store.get(self.document_id))

	def set(self, data, merge: bool = False):
		self.payloads.append({"data": deepcopy(data), "merge": merge})
		if merge and self.document_id in self.store:
			self.store[self.document_id].update(data)
		else:
			self.store[self.document_id] = data


class FakeCollection:
	def __init__(self, store: dict):
		self.store = store
		self.documents = {}

	def document(self, document_id: str):
		if document_id not in self.documents:
			self.documents[document_id] = FakeDocument(self.store, document_id)

		return self.documents[document_id]


class FakeDb:
	def __init__(self):
		self.collections = {
			"submissions": {},
			"results": {},
		}
		self.collection_refs = {}

	def collection(self, name: str):
		if name not in self.collection_refs:
			self.collection_refs[name] = FakeCollection(self.collections[name])

		return self.collection_refs[name]

	def result_document(self, submission_id: str):
		return self.collection("results").document(submission_id)
