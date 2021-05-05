# Tests to run
SET( EXAMPLE_SOURCES HelloSO3)

FOREACH(example_src ${EXAMPLE_SOURCES})
  ADD_EXECUTABLE( ${example_src} ${example_src}.cpp)
  TARGET_LINK_LIBRARIES( ${example_src} sophus fmt::fmt)
ENDFOREACH(example_src)
