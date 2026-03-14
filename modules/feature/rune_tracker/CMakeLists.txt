VisCore_add_module(rune_tracker
DEPENDS rune_combo)

# Ensure cmake re-runs when new source files are added to or removed from src/,
# so that file(GLOB_RECURSE) in VisCoreModule.cmake always picks up all sources.
set_property(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}" APPEND PROPERTY
    CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src")
 