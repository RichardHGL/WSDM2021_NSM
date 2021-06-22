# https://github.com/microsoft/FastRDFStore/blob/master/FastRDFStore/FastRDFStore.cs
# code reference
from struct import *


class BinaryStream:
    def __init__(self, base_stream):
        self.base_stream = base_stream

    def readByte(self):
        return self.base_stream.read(1)

    def readBytes(self, length):
        return self.base_stream.read(length)

    def readChar(self):
        return self.unpack('b')

    def readUChar(self):
        return self.unpack('B')

    def readBool(self):
        return self.unpack('?')

    def readInt16(self):
        return self.unpack('h', 2)

    def readUInt16(self):
        return self.unpack('H', 2)

    def readInt32(self):
        return self.unpack('i', 4)

    def readUInt32(self):
        return self.unpack('I', 4)

    def readInt64(self):
        return self.unpack('q', 8)

    def readUInt64(self):
        return self.unpack('Q', 8)

    def readFloat(self):
        return self.unpack('f', 4)

    def readDouble(self):
        return self.unpack('d', 8)

    def readString(self):
        # length = self.readUInt16()
        length = self.unpack('<B')
        return self.unpack(str(length) + 's', length)

    def writeBytes(self, value):
        self.base_stream.write(value)

    def writeChar(self, value):
        self.pack('c', value)

    def writeUChar(self, value):
        self.pack('C', value)

    def writeBool(self, value):
        self.pack('?', value)

    def writeInt16(self, value):
        self.pack('h', value)

    def writeUInt16(self, value):
        self.pack('H', value)

    def writeInt32(self, value):
        self.pack('i', value)

    def writeUInt32(self, value):
        self.pack('I', value)

    def writeInt64(self, value):
        self.pack('q', value)

    def writeUInt64(self, value):
        self.pack('Q', value)

    def writeFloat(self, value):
        self.pack('f', value)

    def writeDouble(self, value):
        self.pack('d', value)

    def writeString(self, value):
        length = len(value)
        self.writeUInt16(length)
        self.pack(str(length) + 's', value)

    def pack(self, fmt, data):
        return self.writeBytes(pack(fmt, data))

    def unpack(self, fmt, length = 1):
        return unpack(fmt, self.readBytes(length))[0]


def get_key(subject):
    if subject.startswith("m.") or subject.startswith("g."):
        if len(subject) > 3:
            return subject[0:4]
        elif len(subject) > 2:
            return subject[0:3]
        else:
            return subject[0:2]
    else:
        if len(subject) > 1:
            return subject[0:2]
        return subject[0:1]


def is_cvt(subject, cvt_nodes):
    tp_key = get_key(subject)
    if tp_key in cvt_nodes:
        if subject in cvt_nodes[tp_key]:
            return cvt_nodes[tp_key][subject]
    return False


def load_cvt():
    # filename = "/mnt/DGX-1-Vol01/gaolehe/data/freebase_webqsp/data/cvtnodes.bin"
    filename = "/home/hegaole/data/temp/freebase_webqsp/data/cvtnodes.bin"
    f = open(filename, "rb")
    reader = BinaryStream(f)
    dictionariesCount = reader.readInt32()
    # print(dictionariesCount)
    to_return = {}
    for i in range(0, dictionariesCount):
        key = bytes.decode(reader.readString())
        # covert byte to string
        count = reader.readInt32()
        # print(key, count)
        dict_tp = {}
        for j in range(0, count):
            mid = bytes.decode(reader.readString())
            isCVT = reader.readBool()
            dict_tp[mid] = isCVT
        to_return[key] = dict_tp
    return to_return


if __name__ == "__main__":
    to_return = load_cvt()
    tp_dict = to_return["m.01"]
    num = 0
    for item in tp_dict:
        print(item, tp_dict[item], type(tp_dict[item]))
        num += 1
        if num == 10:
            exit(-1)