/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

%ignore "";

%rename("%s") TFE_NewContext;
%rename("%s") TFE_DeleteContext;
%rename("%s") TFE_ContextListDevices;
%rename("%s") TFE_ContextAddFunctionDef;
%rename("%s") TFE_NewOp;
%rename("%s") TFE_OpGetAttrType;
%rename("%s") TFE_Py_InitEagerTensor;
%rename("%s") TFE_Py_RegisterExceptionClass;
%rename("%s") TFE_Py_Execute;
%rename("%s") TFE_Py_UID;


%{
#include "tensorflow/python/eager/pywrap_tfe.h"
%}

%typemap(out) TF_DataType {
  $result = PyInt_FromLong($1);
}

%typemap(out) int64_t {
  $result = PyInt_FromLong($1);
}

%typemap(out) TF_AttrType {
  $result = PyInt_FromLong($1);
}

%typemap(in, numinputs=0) unsigned char* is_list (unsigned char tmp) {
  $1 = &tmp;
}

%typemap(argout) unsigned char* is_list {
  if (*$1 == 1) {
    PyObject* list = PyList_New(1);
    PyList_SetItem(list, 0, $result);
    $result = list;
  }
}

%typemap(in) const char* serialized_function_def {
  $1 = TFE_GetPythonString($input);
}

%typemap(in) const char* device_name {
  if ($input == Py_None) {
    $1 = nullptr;
  } else {
    $1 = TFE_GetPythonString($input);
  }
}

%typemap(in) const char* op_name {
  $1 = TFE_GetPythonString($input);
}

%typemap(in) (TFE_Context*) {
  $1 = (TFE_Context*)PyCapsule_GetPointer($input, nullptr);

}
%typemap(out) (TFE_Context*) {
  if ($1 == nullptr) {
    SWIG_fail;
  } else {
    $result = PyCapsule_New($1, nullptr, TFE_DeleteContextCapsule);
  }
}

%include "tensorflow/c/eager/c_api.h"

%typemap(in) TFE_InputTensorHandles* inputs (TFE_InputTensorHandles temp) {
  $1 = &temp;
  if ($input != Py_None) {
    if (!PyList_Check($input)) {
      SWIG_exception_fail(SWIG_TypeError,
                          "must provide a list of Tensors as inputs");
    }
    Py_ssize_t len = PyList_Size($input);
    $1->resize(len);
    for (Py_ssize_t i = 0; i < len; ++i) {
      PyObject* elem = PyList_GetItem($input, i);
      if (!elem) {
        SWIG_fail;
      }
      if (EagerTensor_CheckExact(elem)) {
        (*$1)[i] = EagerTensorHandle(elem);
      } else {
        SWIG_exception_fail(SWIG_TypeError,
                            "provided list of inputs contains objects other "
                            "than 'EagerTensor'");
      }
    }
  }
}

// Temporary for the argout
%typemap(in) TFE_OutputTensorHandles* outputs (TFE_OutputTensorHandles temp) {
  if (!PyInt_Check($input)) {
    SWIG_exception_fail(SWIG_TypeError,
                        "expected an integer value (size of the number of "
                        "outputs of the operation)");
  }
  $1 = &temp;
  $1->resize(PyInt_AsLong($input), nullptr);
}

// Create new Status object.
%typemap(in, numinputs=0) TF_Status *out_status {
  $1 = TF_NewStatus();
}

%typemap(freearg) (TF_Status* out_status) {
 TF_DeleteStatus($1);
}

%typemap(argout) (TFE_OutputTensorHandles* outputs, TF_Status* out_status) {
  if (MaybeRaiseExceptionFromTFStatus($2, nullptr)) {
    SWIG_fail;
  } else {
    int num_outputs = $1->size();
    $result = PyList_New(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      PyObject *output;
      output = EagerTensorFromHandle($1->at(i));
      PyList_SetItem($result, i, output);
    }
  }
}


%include "tensorflow/python/eager/pywrap_tfe.h"


// Clear all typemaps.
%typemap(out) TF_DataType;
%typemap(out) int64_t;
%typemap(out) TF_AttrType;
%typemap(in, numinputs=0) TF_Status *out_status;
%typemap(argout) unsigned char* is_list;
%typemap(in) (TFE_Context*);
%typemap(out) (TFE_Context*);
%typemap(in) TFE_OutputTensorHandles* outputs (TFE_OutputTensorHandles temp);
%typemap(in, numinputs=0) TF_Status *out_status;
%typemap(freearg) (TF_Status* out_status);
%typemap(argout) (TFE_OutputTensorHandles* outputs, TF_Status* out_status);
