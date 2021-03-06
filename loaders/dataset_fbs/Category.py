# automatically generated by the FlatBuffers compiler, do not modify

# namespace: DLFS

import flatbuffers

class Category(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsCategory(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Category()
        x.Init(buf, n + offset)
        return x

    # Category
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Category
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Category
    def Id(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint16Flags, o + self._tab.Pos)
        return 0

    # Category
    def Examples(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    # Category
    def ExamplesAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint64Flags, o)
        return 0

    # Category
    def ExamplesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Category
    def NumImages(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

    # Category
    def NumAnns(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

def CategoryStart(builder): builder.StartObject(5)
def CategoryAddName(builder, name): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)
def CategoryAddId(builder, id): builder.PrependUint16Slot(1, id, 0)
def CategoryAddExamples(builder, examples): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(examples), 0)
def CategoryStartExamplesVector(builder, numElems): return builder.StartVector(8, numElems, 8)
def CategoryAddNumImages(builder, numImages): builder.PrependUint64Slot(3, numImages, 0)
def CategoryAddNumAnns(builder, numAnns): builder.PrependUint64Slot(4, numAnns, 0)
def CategoryEnd(builder): return builder.EndObject()
