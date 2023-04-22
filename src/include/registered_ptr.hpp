#ifndef MSCCLPP_REGISTERED_PTR_HPP_
#define MSCCLPP_REGISTERED_PTR_HPP_

namespace mscclpp {

template<typename T>
class RegisteredPtr {
  RegisteredMemory memory;
  size_t offset;
public:
  RegisteredPtr(RegisteredMemory memory, size_t offset) : memory(memory), offset(offset) {}
  RegisteredPtr(RegisteredMemory memory) : RegisteredPtr(memory, 0) {}
  ~RegisteredPtr() {}

  RegisteredMemory memory() {
    return memory;
  }

  T* data() {
    return reinterpret_cast<T*>(memory.data());
  }

  size_t size() {
    return memory.size() / sizeof(T);
  }

  size_t offset() {
    return offset;
  }

  RegisteredPtr<T> operator+(size_t offset) {
    return RegisteredPtr<T>(memory, this->offset + offset);
  }

  // TODO: all other relevant overloads
};

} // namespace mscclpp

#endif // MSCCLPP_REGISTERED_PTR_HPP_