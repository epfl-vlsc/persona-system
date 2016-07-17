/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/python/client/tf_session_helper.h"

#include <cstring>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/equal_graph_def.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tf_status_helper.h"

namespace tensorflow {

namespace {

// Container types for the various temporary values used internally in
// the wrapper.

// A TF_TensorVector is a vector of borrowed pointers to TF_Tensors.
typedef gtl::InlinedVector<TF_Tensor*, 8> TF_TensorVector;

// Safe containers for (an) owned TF_Tensor(s). On destruction, the
// tensor will be deleted by TF_DeleteTensor.
typedef std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)>
    Safe_TF_TensorPtr;
typedef std::vector<Safe_TF_TensorPtr> Safe_TF_TensorVector;
Safe_TF_TensorPtr make_safe(TF_Tensor* tensor) {
  return Safe_TF_TensorPtr(tensor, TF_DeleteTensor);
}

Status PyArrayDescr_to_TF_DataType(PyArray_Descr* descr,
                                   TF_DataType* out_tf_datatype) {
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  if (PyDict_Next(descr->fields, &pos, &key, &value)) {
    const char* key_string = PyBytes_AsString(key);
    if (!key_string) {
      return errors::Internal("Corrupt numpy type descriptor");
    }
    tensorflow::string key = key_string;
    // The typenames here should match the field names in the custom struct
    // types constructed in test_util.py.
    // TODO(mrry,keveman): Investigate Numpy type registration to replace this
    // hard-coding of names.
    if (key == "quint8") {
      *out_tf_datatype = TF_QUINT8;
    } else if (key == "qint8") {
      *out_tf_datatype = TF_QINT8;
    } else if (key == "qint16") {
      *out_tf_datatype = TF_QINT16;
    } else if (key == "quint16") {
      *out_tf_datatype = TF_QUINT16;
    } else if (key == "qint32") {
      *out_tf_datatype = TF_QINT32;
    } else {
      return errors::Internal("Unsupported numpy data type");
    }
    return Status::OK();
  }
  return errors::Internal("Unsupported numpy data type");
}

Status PyArray_TYPE_to_TF_DataType(PyArrayObject* array,
                                   TF_DataType* out_tf_datatype) {
  int pyarray_type = PyArray_TYPE(array);
  PyArray_Descr* descr = PyArray_DESCR(array);
  switch (pyarray_type) {
    case NPY_FLOAT16:
      *out_tf_datatype = TF_HALF;
      break;
    case NPY_FLOAT32:
      *out_tf_datatype = TF_FLOAT;
      break;
    case NPY_FLOAT64:
      *out_tf_datatype = TF_DOUBLE;
      break;
    case NPY_INT32:
      *out_tf_datatype = TF_INT32;
      break;
    case NPY_UINT8:
      *out_tf_datatype = TF_UINT8;
      break;
    case NPY_UINT16:
      *out_tf_datatype = TF_UINT16;
      break;
    case NPY_INT8:
      *out_tf_datatype = TF_INT8;
      break;
    case NPY_INT16:
      *out_tf_datatype = TF_INT16;
      break;
    case NPY_INT64:
      *out_tf_datatype = TF_INT64;
      break;
    case NPY_BOOL:
      *out_tf_datatype = TF_BOOL;
      break;
    case NPY_COMPLEX64:
      *out_tf_datatype = TF_COMPLEX64;
      break;
    case NPY_COMPLEX128:
      *out_tf_datatype = TF_COMPLEX128;
      break;
    case NPY_OBJECT:
      *out_tf_datatype = TF_STRING;
      break;
    case NPY_VOID:
      // Quantized types are currently represented as custom struct types.
      // PyArray_TYPE returns NPY_VOID for structs, and we should look into
      // descr to derive the actual type.
      return PyArrayDescr_to_TF_DataType(descr, out_tf_datatype);
    default:
      // TODO(mrry): Support these.
      return errors::Internal("Unsupported feed type");
  }
  return Status::OK();
}

Status TF_DataType_to_PyArray_TYPE(TF_DataType tf_datatype,
                                   int* out_pyarray_type) {
  switch (tf_datatype) {
    case TF_HALF:
      *out_pyarray_type = NPY_FLOAT16;
      break;
    case TF_FLOAT:
      *out_pyarray_type = NPY_FLOAT32;
      break;
    case TF_DOUBLE:
      *out_pyarray_type = NPY_FLOAT64;
      break;
    case TF_INT32:
      *out_pyarray_type = NPY_INT32;
      break;
    case TF_UINT8:
      *out_pyarray_type = NPY_UINT8;
      break;
    case TF_UINT16:
      *out_pyarray_type = NPY_UINT16;
      break;
    case TF_INT8:
      *out_pyarray_type = NPY_INT8;
      break;
    case TF_INT16:
      *out_pyarray_type = NPY_INT16;
      break;
    case TF_INT64:
      *out_pyarray_type = NPY_INT64;
      break;
    case TF_BOOL:
      *out_pyarray_type = NPY_BOOL;
      break;
    case TF_COMPLEX64:
      *out_pyarray_type = NPY_COMPLEX64;
      break;
    case TF_COMPLEX128:
      *out_pyarray_type = NPY_COMPLEX128;
      break;
    case TF_STRING:
      *out_pyarray_type = NPY_OBJECT;
      break;
    // TODO(keveman): These should be changed to NPY_VOID, and the type used for
    // the resulting numpy array should be the custom struct types that we
    // expect for quantized types.
    case TF_QINT8:
      *out_pyarray_type = NPY_INT8;
      break;
    case TF_QUINT8:
      *out_pyarray_type = NPY_UINT8;
      break;
    case TF_QINT16:
      *out_pyarray_type = NPY_INT16;
      break;
    case TF_QUINT16:
      *out_pyarray_type = NPY_UINT16;
      break;
    case TF_QINT32:
      *out_pyarray_type = NPY_INT32;
      break;
    case TF_BFLOAT16:
      *out_pyarray_type = NPY_UINT16;
      break;
    default:
      return errors::Internal("Unsupported fetch type");
  }
  return Status::OK();
}

// Iterate over the string array 'array', extract the ptr and len of each string
// element and call f(ptr, len).
template <typename F>
Status PyBytesArrayMap(PyArrayObject* array, F f) {
  Safe_PyObjectPtr iter = tensorflow::make_safe(
      PyArray_IterNew(reinterpret_cast<PyObject*>(array)));
  while (PyArray_ITER_NOTDONE(iter.get())) {
    auto item = tensorflow::make_safe(PyArray_GETITEM(
        array, static_cast<char*>(PyArray_ITER_DATA(iter.get()))));
    if (!item.get()) {
      return errors::Internal("Unable to get element from the feed.");
    }
    char* ptr;
    Py_ssize_t len;

#if PY_VERSION_HEX >= 0x03030000
    // Accept unicode in Python 3, by converting to UTF-8 bytes.
    if (PyUnicode_Check(item.get())) {
      ptr = PyUnicode_AsUTF8AndSize(item.get(), &len);
      if (!ptr) {
        return errors::Internal("Unable to get element from the feed.");
      }
    } else {
#endif
      int success = PyBytes_AsStringAndSize(item.get(), &ptr, &len);
      if (success != 0) {
        return errors::Internal("Unable to get element from the feed.");
      }
#if PY_VERSION_HEX >= 0x03030000
    }
#endif
    f(ptr, len);
    PyArray_ITER_NEXT(iter.get());
  }
  return Status::OK();
}

// Encode the strings in 'array' into a contiguous buffer and return the base of
// the buffer. The caller takes ownership of the buffer.
Status EncodePyBytesArray(PyArrayObject* array, tensorflow::int64 nelems,
                          size_t* size, void** buffer) {
  // Compute bytes needed for encoding.
  *size = 0;
  TF_RETURN_IF_ERROR(PyBytesArrayMap(array, [&size](char* ptr, Py_ssize_t len) {
    *size +=
        sizeof(tensorflow::uint64) + tensorflow::core::VarintLength(len) + len;
  }));
  // Encode all strings.
  std::unique_ptr<char[]> base_ptr(new char[*size]);
  char* base = base_ptr.get();
  char* data_start = base + sizeof(tensorflow::uint64) * nelems;
  char* dst = data_start;  // Where next string is encoded.
  tensorflow::uint64* offsets = reinterpret_cast<tensorflow::uint64*>(base);

  TF_RETURN_IF_ERROR(PyBytesArrayMap(
      array, [&base, &data_start, &dst, &offsets](char* ptr, Py_ssize_t len) {
        *offsets = (dst - data_start);
        offsets++;
        dst = tensorflow::core::EncodeVarint64(dst, len);
        memcpy(dst, ptr, len);
        dst += len;
      }));
  CHECK_EQ(dst, base + *size);
  *buffer = base_ptr.release();
  return Status::OK();
}

// Determine the pointer and offset of the string at offset 'i' in the string
// tensor 'src', whose total length is 'num_elements'.
static Status TF_StringTensor_GetPtrAndLen(const TF_Tensor* src,
                                           tensorflow::int64 num_elements,
                                           tensorflow::int64 i,
                                           const char** ptr,
                                           tensorflow::uint64* len) {
  const char* input = reinterpret_cast<const char*>(TF_TensorData(src));
  const size_t src_size = TF_TensorByteSize(src);
  const char* data_start = input + sizeof(tensorflow::uint64) * num_elements;
  const char* limit = input + src_size;
  tensorflow::uint64 offset =
      reinterpret_cast<const tensorflow::uint64*>(input)[i];
  const char* p =
      tensorflow::core::GetVarint64Ptr(data_start + offset, limit, len);
  if (static_cast<int64>(offset) >= (limit - data_start) || !p ||
      static_cast<int64>(*len) > (limit - p)) {
    return errors::InvalidArgument("Malformed TF_STRING tensor; element ", i,
                                   " out of range");
  }
  *ptr = p;
  return Status::OK();
}

// Copy the string at offset 'i' in the (linearized) string tensor 'tensor' into
// 'pyarray' at offset pointed by the 'i_ptr' iterator.
static Status CopyStringToPyArrayElement(PyArrayObject* pyarray, void* i_ptr,
                                         TF_Tensor* tensor,
                                         tensorflow::int64 num_elements,
                                         tensorflow::int64 i) {
  const char* ptr = nullptr;
  tensorflow::uint64 len = 0;
  TF_RETURN_IF_ERROR(
      TF_StringTensor_GetPtrAndLen(tensor, num_elements, i, &ptr, &len));
  auto py_string = tensorflow::make_safe(PyBytes_FromStringAndSize(ptr, len));
  int success = PyArray_SETITEM(
      pyarray, static_cast<char*>(PyArray_ITER_DATA(i_ptr)), py_string.get());
  if (success != 0) {
    return errors::Internal("Error setting element ", i);
  }
  return Status::OK();
}

// Converts the given TF_Tensor to a Numpy array.
// If the returned status is OK, the caller becomes the owner of *out_array.
Status TF_Tensor_to_PyObject(TF_Tensor* tensor, PyObject** out_array) {
  // A fetched operation will correspond to a null tensor, and a None
  // in Python.
  if (tensor == nullptr) {
    Py_INCREF(Py_None);
    *out_array = Py_None;
    return Status::OK();
  }

  const int ndims = TF_NumDims(tensor);
  gtl::InlinedVector<npy_intp, 4> dims(ndims);
  tensorflow::int64 nelems = 1;
  for (int i = 0; i < ndims; ++i) {
    dims[i] = TF_Dim(tensor, i);
    nelems *= dims[i];
  }

  // Convert TensorFlow dtype to numpy type descriptor.
  int type_num = -1;
  TF_RETURN_IF_ERROR(
      TF_DataType_to_PyArray_TYPE(TF_TensorType(tensor), &type_num));
  PyArray_Descr* descr = PyArray_DescrFromType(type_num);

  // Copy the TF_TensorData into a newly-created ndarray and return it.
  // TODO(mrry): Perhaps investigate zero-copy approaches. This would involve
  // creating an ndarray-like object that wraps the TF_Tensor buffer, and
  // maps its destructor to TF_DeleteTensor.
  Safe_PyObjectPtr safe_out_array =
      tensorflow::make_safe(PyArray_Empty(ndims, dims.data(), descr, 0));
  if (!safe_out_array) {
    return errors::Internal("Could not allocate ndarray");
  }
  PyArrayObject* py_array =
      reinterpret_cast<PyArrayObject*>(safe_out_array.get());
  if (PyArray_NBYTES(py_array) !=
      static_cast<int64>(TF_TensorByteSize(tensor))) {
    if (TF_TensorType(tensor) == TF_STRING) {
      // Copy element by element.
      auto iter = tensorflow::make_safe(PyArray_IterNew(safe_out_array.get()));
      for (tensorflow::int64 i = 0; i < nelems; ++i) {
        auto s =
            CopyStringToPyArrayElement(py_array, iter.get(), tensor, nelems, i);
        if (!s.ok()) {
          return s;
        }
        PyArray_ITER_NEXT(iter.get());
      }
    } else {
      return errors::Internal("ndarray was ", PyArray_NBYTES(py_array),
                              " bytes but TF_Tensor was ",
                              TF_TensorByteSize(tensor), " bytes");
    }
  } else {
    memcpy(PyArray_DATA(py_array), TF_TensorData(tensor),
           PyArray_NBYTES(py_array));
  }

  // PyArray_Return turns rank 0 arrays into numpy scalars
  *out_array = PyArray_Return(
      reinterpret_cast<PyArrayObject*>(safe_out_array.release()));
  return Status::OK();
}

}  // namespace

Safe_PyObjectPtr make_safe(PyObject* o) {
  return Safe_PyObjectPtr(o, Py_DECREF_wrapper);
}

void TF_Run_wrapper_helper(TF_Session* session, const char* handle,
                           const TF_Buffer* run_options,
                           const FeedVector& inputs,
                           const NameVector& output_names,
                           const NameVector& target_nodes,
                           TF_Status* out_status, PyObjectVector* out_values,
                           TF_Buffer* run_outputs) {
  // 1. Convert the feed inputs to the appropriate form for TF_Run.
  NameVector input_names;
  Safe_PyObjectVector
      py_inputs_safe;  // Used to decref the input arrays on failure.
  Safe_TF_TensorVector inputs_safe;  // Used to delete tensors on failure.
  TF_TensorVector inputs_unsafe;     // Used to contain the arg to TF_Run.

  for (const auto& name_and_array : inputs) {
    py_inputs_safe.emplace_back(
        make_safe(reinterpret_cast<PyObject*>(name_and_array.second)));
  }

  Status result;
  for (size_t i = 0; i < inputs.size(); ++i) {
    input_names.push_back(inputs[i].first);
    PyArrayObject* array = inputs[i].second;

    // Convert numpy dtype to TensorFlow dtype.
    TF_DataType dtype = TF_FLOAT;
    result = PyArray_TYPE_to_TF_DataType(array, &dtype);
    if (!result.ok()) {
      Set_TF_Status_from_Status(out_status, result);
      return;
    }

    tensorflow::int64 nelems = 1;
    gtl::InlinedVector<tensorflow::int64, 4> dims;
    for (int i = 0; i < PyArray_NDIM(array); ++i) {
      dims.push_back(PyArray_SHAPE(array)[i]);
      nelems *= dims[i];
    }

    // Create a TF_Tensor based on the fed data. In the case of non-string data
    // type, this steals a reference to array, which will be relinquished when
    // the underlying buffer is deallocated. For string, a new temporary buffer
    // is allocated into which the strings are encoded.
    if (dtype != TF_STRING) {
      // NOTE(mrry): We currently copy the numpy array into a new
      // buffer to avoid possible issues on deallocation (such as
      // having to acquire the Python Global Interpreter Lock).
      // TODO(mrry): Investigate in what cases we can safely acquire
      size_t size = PyArray_NBYTES(array);
      // NOTE(mrry): 32 is the upper bound on current alignment
      // requirements for tensorflow::Tensor. We hard code this here to
      // avoid taking a dependency on Eigen in the client code.
      void* data = tensorflow::cpu_allocator()->AllocateRaw(32, size);
      if (tensorflow::LogMemory::IsEnabled()) {
        LogMemory::RecordRawAllocation(
            "Python session helper",
            tensorflow::LogMemory::EXTERNAL_TENSOR_ALLOCATION_STEP_ID, size,
            data, tensorflow::cpu_allocator());
      }
      std::memcpy(data, PyArray_DATA(array), size);
      inputs_safe.emplace_back(make_safe(TF_NewTensor(
          dtype, dims.data(), dims.size(), data, size,
          [](void* data, size_t len, void* arg) {
            if (tensorflow::LogMemory::IsEnabled()) {
              LogMemory::RecordRawDeallocation(
                  "Python session helper",
                  tensorflow::LogMemory::EXTERNAL_TENSOR_ALLOCATION_STEP_ID,
                  data, tensorflow::cpu_allocator(), false);
            }
            tensorflow::cpu_allocator()->DeallocateRaw(data);
          },
          nullptr)));
      // The destruction of the numpy array will now be handled by the
      // inputs_safe destructor.
      py_inputs_safe[i].reset();
    } else {
      size_t size = 0;
      void* encoded = nullptr;
      Status s = EncodePyBytesArray(array, nelems, &size, &encoded);
      if (!s.ok()) {
        Set_TF_Status_from_Status(out_status, s);
        return;
      }
      inputs_safe.emplace_back(
          make_safe(TF_NewTensor(dtype, dims.data(), dims.size(), encoded, size,
                                 [](void* data, size_t len, void* arg) {
                                   delete[] reinterpret_cast<char*>(data);
                                 },
                                 array)));
      // The destruction of the numpy array will now be handled by the
      // inputs_safe destructor.
      py_inputs_safe[i].reset();
    }
    inputs_unsafe.push_back(inputs_safe.back().get());
  }

  // 2. Allocate a container for the output data.
  TF_TensorVector outputs(output_names.size());

  // 3. Actually call TF_Run().
  Py_BEGIN_ALLOW_THREADS;
  if (handle == nullptr) {
    TF_Run(session, run_options, input_names.data(), inputs_unsafe.data(),
           input_names.size(), const_cast<const char**>(output_names.data()),
           outputs.data(), output_names.size(),
           const_cast<const char**>(target_nodes.data()), target_nodes.size(),
           run_outputs, out_status);
  } else {
    TF_PRun(session, handle, input_names.data(), inputs_unsafe.data(),
            input_names.size(), const_cast<const char**>(output_names.data()),
            outputs.data(), output_names.size(),
            const_cast<const char**>(target_nodes.data()), target_nodes.size(),
            out_status);
  }

  Py_END_ALLOW_THREADS;

  // 4. The TensorFlow runtime has taken ownership of the fed tensors,
  // so we release the safe pointers to them.
  for (auto& input : inputs_safe) {
    input.release();
  }

  if (TF_GetCode(out_status) != TF_OK) {
    return;
  }

  // 5. We now own the fetched tensors, so set up a safe container to
  // delete them when we exit this scope.
  Safe_TF_TensorVector tf_outputs_safe;
  for (const auto& output : outputs) {
    tf_outputs_safe.emplace_back(make_safe(output));
  }

  // 6. Convert the fetched tensors into numpy ndarrays. Store them in a safe
  // container so that we do not leak
  Safe_PyObjectVector py_outputs_safe;
  for (size_t i = 0; i < output_names.size(); ++i) {
    PyObject* py_array;
    result = TF_Tensor_to_PyObject(outputs[i], &py_array);
    if (!result.ok()) {
      Set_TF_Status_from_Status(out_status, result);
      return;
    }
    py_outputs_safe.emplace_back(make_safe(py_array));
  }

  // 7. If we reach this point, we have successfully built a list of objects
  // so we can release them from the safe container.
  for (auto& output : py_outputs_safe) {
    out_values->push_back(output.release());
  }
}

// Wrapper for TF_Run that converts the arguments to appropriate types.
// If *out_status is OK, the caller becomes the owner of the PyObjects
// in *out_values.
void TF_Run_wrapper(TF_Session* session, const TF_Buffer* run_options,
                    const FeedVector& inputs, const NameVector& output_names,
                    const NameVector& target_nodes, TF_Status* out_status,
                    PyObjectVector* out_values, TF_Buffer* run_outputs) {
  TF_Run_wrapper_helper(session, nullptr, run_options, inputs, output_names,
                        target_nodes, out_status, out_values, run_outputs);
}

// Wrapper for TF_PRunSetup that converts the arguments to appropriate types.
// If *out_status is OK, the caller becomes the owner of *out_handle.
void TF_PRunSetup_wrapper(TF_Session* session, const NameVector& input_names,
                          const NameVector& output_names,
                          const NameVector& target_nodes, TF_Status* out_status,
                          char** out_handle) {
  Py_BEGIN_ALLOW_THREADS;
  TF_PRunSetup(
      session, const_cast<const char**>(input_names.data()), input_names.size(),
      const_cast<const char**>(output_names.data()), output_names.size(),
      const_cast<const char**>(target_nodes.data()), target_nodes.size(),
      out_handle, out_status);
  Py_END_ALLOW_THREADS;
}

// Wrapper for TF_PRun that converts the arguments to appropriate types.
// If *out_status is OK, the caller becomes the owner of the PyObjects
// in *out_values.
void TF_PRun_wrapper(TF_Session* session, const char* handle,
                     const FeedVector& inputs, const NameVector& output_names,
                     TF_Status* out_status, PyObjectVector* out_values) {
  TF_Run_wrapper_helper(session, handle, nullptr, inputs, output_names,
                        NameVector(), out_status, out_values, nullptr);
}

string EqualGraphDefWrapper(const string& actual, const string& expected) {
  GraphDef actual_def;
  if (!actual_def.ParseFromString(actual)) {
    return "actual is not a valid serialized GraphDef";
  }
  GraphDef expected_def;
  if (!expected_def.ParseFromString(expected)) {
    return "expected is not a valid serialized GraphDef";
  }
  string diff;
  return EqualGraphDef(actual_def, expected_def, &diff) ? "" : diff;
}

}  // namespace tensorflow
