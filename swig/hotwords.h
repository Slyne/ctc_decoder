// Copyright (c) 2023, 58.com(Wuba) Inc AI Lab. All Rights Reserved.
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
// Modified from DeepSpeech(https://github.com/mozilla/DeepSpeech)

#ifndef HOTWORDS_H
#define HOTWORDS_H

#include <iostream>
#include <string>
#include <unordered_map>

#include "scorer.h"

class HotWordsBoosting {
  public:
    HotWordsBoosting(const std::unordered_map<std::string, float> &hotwords_dict, const std::vector<std::string>& char_list,
                     int window_length=4, int SPACE_ID=-1, bool is_character_based=true);
    ~HotWordsBoosting();

    // make ngram for a given prefix
    std::pair<int, std::vector<std::string>> make_ngram(PathTrie *prefix);

    // translate the vector in index to string
    std::string vec2str(const std::vector<int>& input);

    // add hotwords score
    float get_hotwords_score(const std::vector<std::string>& words, int begin_index);

    std::unordered_map<std::string, float> hotwords_dict;
    int window_length;
    int SPACE_ID;
    bool is_character_based;
    std::vector<std::string> char_list;
};

#endif  // HOTWORDS_H