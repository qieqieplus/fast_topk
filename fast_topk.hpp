// This fast top-k library adopts code from ScaNN
// (https://github.com/google-research/google-research/tree/master/scann)
//
// Copyright 2024 The Google Research Authors.
//
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


#ifndef FAST_TOPK_HPP_
#define FAST_TOPK_HPP_

#include "internal/span.hpp"
#include "internal/topk.hpp"

namespace fast_topk {

using internal::MutableSpan;
using internal::ConstSpan;
using internal::MakeSpan;
using internal::MakeConstSpan;
using internal::TopNeighbors;

}

#endif //FAST_TOPK_HPP_
