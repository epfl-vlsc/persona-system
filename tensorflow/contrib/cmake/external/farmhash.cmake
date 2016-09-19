include (ExternalProject)

set(farmhash_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/farmhash_archive)
#set(farmhash_EXTRA_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/farmhash/src)
set(farmhash_URL https://github.com/google/farmhash/archive/34c13ddfab0e35422f4c3979f360635a8c050260.zip)
set(farmhash_HASH SHA256=e3d37a59101f38fd58fb799ed404d630f0eee18bfc2a2433910977cc8fea9c28)
set(farmhash_BUILD ${CMAKE_BINARY_DIR}/farmhash/src/farmhash)
set(farmhash_INSTALL ${CMAKE_BINARY_DIR}/farmhash/install)
#set(farmhash_LIBRARIES ${farmhash_BUILD}/obj/so/libfarmhash.so)
set(farmhash_STATIC_LIBRARIES
    ${farmhash_INSTALL}/lib/libfarmhash.a
)
set(farmhash_INCLUDES ${farmhash_BUILD})

set(farmhash_HEADERS
    "${farmhash_BUILD}/src/farmhash.h"
)

ExternalProject_Add(farmhash
    PREFIX farmhash
    URL ${farmhash_URL}
    URL_HASH ${farmhash_HASH}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_COMMAND $(MAKE)
    INSTALL_COMMAND $(MAKE) install
    CONFIGURE_COMMAND
        ${farmhash_BUILD}/configure
        --prefix=${farmhash_INSTALL}
        --enable-shared=yes
)

# put farmhash includes in the directory where they are expected
add_custom_target(farmhash_create_destination_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${farmhash_INCLUDE_DIR}/farmhash-34c13ddfab0e35422f4c3979f360635a8c050260/src
    DEPENDS farmhash)

add_custom_target(farmhash_copy_headers_to_destination
    DEPENDS farmhash_create_destination_dir)

foreach(header_file ${farmhash_HEADERS})
    add_custom_command(TARGET farmhash_copy_headers_to_destination PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${header_file} ${farmhash_INCLUDE_DIR}/farmhash-34c13ddfab0e35422f4c3979f360635a8c050260/src)
endforeach()
