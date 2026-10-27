#pragma once

// Optional device tables indexed by ORIGINAL expert ID. Null tables retain
// the contiguous resident-weight addressing and the same dot-product order.
struct QfnExpertPointers {
  const void* const* first{nullptr};
  const void* const* second{nullptr};
};
