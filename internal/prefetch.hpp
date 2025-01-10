// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef INTERNAL_PREFETCH_H
#define INTERNAL_PREFETCH_H

#include "immintrin.h"

namespace internal {

enum class PrefetchStrategy { kOff, kSeq, kSmart };

enum PrefetchHint {
  PREFETCH_HINT_NTA = _MM_HINT_NTA,
  PREFETCH_HINT_T2 = _MM_HINT_T2,
  PREFETCH_HINT_T1 = _MM_HINT_T1,
  PREFETCH_HINT_T0 = _MM_HINT_T0,
};

template <PrefetchHint hint>
inline void prefetch(const void *address) {
#ifdef _MSC_VER
  // For MSVC, use _mm_prefetch
  _mm_prefetch(address, hint);
#else
  // For GCC/Clang, use __builtin_prefetch
  __builtin_prefetch(address, 0, hint);
#endif
}
}  // namespace internal

#endif  // INTERNAL_PREFETCH_H
