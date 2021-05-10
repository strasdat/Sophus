# Tests to run
SET( EXAMPLE_SOURCES HelloSO3)

FOREACH(example_src ${EXAMPLE_SOURCES})
  ADD_EXECUTABLE( ${example_src} ${example_src}.cpp)
  TARGET_LINK_LIBRARIES( ${example_src} sophus)
  if (NOT USE_BASIC_LOGGING)
    TARGET_LINK_LIBRARIES( ${example_src} fmt::fmt)
  endif()
ENDFOREACH(example_src)
