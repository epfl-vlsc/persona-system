# Class `tensorflow::Env`

An interface used by the tensorflow implementation to access operating system functionality like the filesystem etc.

Callers may wish to provide a custom Env object to get fine grain control.

All Env implementations are safe for concurrent access from multiple threads without any external synchronization.

##Member Summary

* [`tensorflow::Env::Env()`](#tensorflow_Env_Env)
* [`tensorflow::Env::~Env()`](#tensorflow_Env_Env)
* [`virtual Status tensorflow::Env::NewRandomAccessFile(const string &fname, RandomAccessFile **result)=0`](#virtual_Status_tensorflow_Env_NewRandomAccessFile)
  * Creates a brand new random access read-only file with the specified name.
* [`virtual Status tensorflow::Env::NewWritableFile(const string &fname, WritableFile **result)=0`](#virtual_Status_tensorflow_Env_NewWritableFile)
  * Creates an object that writes to a new file with the specified name.
* [`virtual Status tensorflow::Env::NewAppendableFile(const string &fname, WritableFile **result)=0`](#virtual_Status_tensorflow_Env_NewAppendableFile)
  * Creates an object that either appends to an existing file, or writes to a new file (if the file does not exist to begin with).
* [`virtual bool tensorflow::Env::FileExists(const string &fname)=0`](#virtual_bool_tensorflow_Env_FileExists)
  * Returns true iff the named file exists.
* [`virtual Status tensorflow::Env::GetChildren(const string &dir, std::vector< string > *result)=0`](#virtual_Status_tensorflow_Env_GetChildren)
  * Stores in *result the names of the children of the specified directory. The names are relative to "dir".
* [`virtual Status tensorflow::Env::DeleteFile(const string &fname)=0`](#virtual_Status_tensorflow_Env_DeleteFile)
  * Deletes the named file.
* [`virtual Status tensorflow::Env::CreateDir(const string &dirname)=0`](#virtual_Status_tensorflow_Env_CreateDir)
  * Creates the specified directory.
* [`virtual Status tensorflow::Env::DeleteDir(const string &dirname)=0`](#virtual_Status_tensorflow_Env_DeleteDir)
  * Deletes the specified directory.
* [`virtual Status tensorflow::Env::GetFileSize(const string &fname, uint64 *file_size)=0`](#virtual_Status_tensorflow_Env_GetFileSize)
  * Stores the size of `fname` in `*file_size`.
* [`virtual Status tensorflow::Env::RenameFile(const string &src, const string &target)=0`](#virtual_Status_tensorflow_Env_RenameFile)
  * Renames file src to target. If target already exists, it will be replaced.
* [`virtual uint64 tensorflow::Env::NowMicros()=0`](#virtual_uint64_tensorflow_Env_NowMicros)
  * Returns the number of micro-seconds since some fixed point in time. Only useful for computing deltas of time.
* [`virtual void tensorflow::Env::SleepForMicroseconds(int micros)=0`](#virtual_void_tensorflow_Env_SleepForMicroseconds)
  * Sleeps/delays the thread for the prescribed number of micro-seconds.
* [`virtual Thread* tensorflow::Env::StartThread(const ThreadOptions &thread_options, const string &name, std::function< void()> fn) TF_MUST_USE_RESULT=0`](#virtual_Thread_tensorflow_Env_StartThread)
  * Returns a new thread that is running fn() and is identified (for debugging/performance-analysis) by "name".
* [`virtual void tensorflow::Env::SchedClosure(std::function< void()> closure)=0`](#virtual_void_tensorflow_Env_SchedClosure)
* [`virtual void tensorflow::Env::SchedClosureAfter(int micros, std::function< void()> closure)=0`](#virtual_void_tensorflow_Env_SchedClosureAfter)
* [`virtual Status tensorflow::Env::LoadLibrary(const char *library_filename, void **handle)=0`](#virtual_Status_tensorflow_Env_LoadLibrary)
* [`virtual Status tensorflow::Env::GetSymbolFromLibrary(void *handle, const char *symbol_name, void **symbol)=0`](#virtual_Status_tensorflow_Env_GetSymbolFromLibrary)
* [`static Env* tensorflow::Env::Default()`](#static_Env_tensorflow_Env_Default)
  * Returns a default environment suitable for the current operating system.

##Member Details

#### `tensorflow::Env::Env()` {#tensorflow_Env_Env}





#### `tensorflow::Env::~Env()` {#tensorflow_Env_Env}





#### `virtual Status tensorflow::Env::NewRandomAccessFile(const string &fname, RandomAccessFile **result)=0` {#virtual_Status_tensorflow_Env_NewRandomAccessFile}

Creates a brand new random access read-only file with the specified name.

On success, stores a pointer to the new file in *result and returns OK. On failure stores NULL in *result and returns non-OK. If the file does not exist, returns a non-OK status.

The returned file may be concurrently accessed by multiple threads.

#### `virtual Status tensorflow::Env::NewWritableFile(const string &fname, WritableFile **result)=0` {#virtual_Status_tensorflow_Env_NewWritableFile}

Creates an object that writes to a new file with the specified name.

Deletes any existing file with the same name and creates a new file. On success, stores a pointer to the new file in *result and returns OK. On failure stores NULL in *result and returns non-OK.

The returned file will only be accessed by one thread at a time.

#### `virtual Status tensorflow::Env::NewAppendableFile(const string &fname, WritableFile **result)=0` {#virtual_Status_tensorflow_Env_NewAppendableFile}

Creates an object that either appends to an existing file, or writes to a new file (if the file does not exist to begin with).

On success, stores a pointer to the new file in *result and returns OK. On failure stores NULL in *result and returns non-OK.

The returned file will only be accessed by one thread at a time.

#### `virtual bool tensorflow::Env::FileExists(const string &fname)=0` {#virtual_bool_tensorflow_Env_FileExists}

Returns true iff the named file exists.



#### `virtual Status tensorflow::Env::GetChildren(const string &dir, std::vector< string > *result)=0` {#virtual_Status_tensorflow_Env_GetChildren}

Stores in *result the names of the children of the specified directory. The names are relative to "dir".

Original contents of *results are dropped.

#### `virtual Status tensorflow::Env::DeleteFile(const string &fname)=0` {#virtual_Status_tensorflow_Env_DeleteFile}

Deletes the named file.



#### `virtual Status tensorflow::Env::CreateDir(const string &dirname)=0` {#virtual_Status_tensorflow_Env_CreateDir}

Creates the specified directory.



#### `virtual Status tensorflow::Env::DeleteDir(const string &dirname)=0` {#virtual_Status_tensorflow_Env_DeleteDir}

Deletes the specified directory.



#### `virtual Status tensorflow::Env::GetFileSize(const string &fname, uint64 *file_size)=0` {#virtual_Status_tensorflow_Env_GetFileSize}

Stores the size of `fname` in `*file_size`.



#### `virtual Status tensorflow::Env::RenameFile(const string &src, const string &target)=0` {#virtual_Status_tensorflow_Env_RenameFile}

Renames file src to target. If target already exists, it will be replaced.



#### `virtual uint64 tensorflow::Env::NowMicros()=0` {#virtual_uint64_tensorflow_Env_NowMicros}

Returns the number of micro-seconds since some fixed point in time. Only useful for computing deltas of time.



#### `virtual void tensorflow::Env::SleepForMicroseconds(int micros)=0` {#virtual_void_tensorflow_Env_SleepForMicroseconds}

Sleeps/delays the thread for the prescribed number of micro-seconds.



#### `virtual Thread* tensorflow::Env::StartThread(const ThreadOptions &thread_options, const string &name, std::function< void()> fn) TF_MUST_USE_RESULT=0` {#virtual_Thread_tensorflow_Env_StartThread}

Returns a new thread that is running fn() and is identified (for debugging/performance-analysis) by "name".

Caller takes ownership of the result and must delete it eventually (the deletion will block until fn() stops running).

#### `virtual void tensorflow::Env::SchedClosure(std::function< void()> closure)=0` {#virtual_void_tensorflow_Env_SchedClosure}





#### `virtual void tensorflow::Env::SchedClosureAfter(int micros, std::function< void()> closure)=0` {#virtual_void_tensorflow_Env_SchedClosureAfter}





#### `virtual Status tensorflow::Env::LoadLibrary(const char *library_filename, void **handle)=0` {#virtual_Status_tensorflow_Env_LoadLibrary}





#### `virtual Status tensorflow::Env::GetSymbolFromLibrary(void *handle, const char *symbol_name, void **symbol)=0` {#virtual_Status_tensorflow_Env_GetSymbolFromLibrary}





#### `static Env* tensorflow::Env::Default()` {#static_Env_tensorflow_Env_Default}

Returns a default environment suitable for the current operating system.

Sophisticated users may wish to provide their own Env implementation instead of relying on this default environment.

The result of Default() belongs to this library and must never be deleted.
