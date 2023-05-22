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

#include <iostream>
#include <unordered_map>

#include "path_trie.h"
#include "hotwords.h"
#include "scorer.h"

HotWordsScorer::HotWordsScorer(const std::unordered_map<std::string, float> &hotwords_dict, const std::vector<std::string>& char_list,
                                   int window_length, int SPACE_ID, bool is_character_based) {
    this->hotwords_dict = hotwords_dict;
    this->window_length = window_length;
    this->is_character_based = is_character_based;
    this->SPACE_ID = SPACE_ID;
    this->char_list = char_list;
}

HotWordsScorer::~HotWordsScorer(){
}

std::string HotWordsScorer::vec2str(const std::vector<int>& input) {
    std::string word;
    for (auto ind : input) {
        word += this->char_list[ind];
    }
    return word;
}

std::pair<int, std::vector<std::string>> HotWordsScorer::make_ngram(PathTrie* prefix){
    std::vector<std::string> ngram;
    PathTrie* current_node = prefix;
    PathTrie* new_node = nullptr;
    int no_start_token_count = 0;
    for (int order = 0; order < this->window_length; order++) {
        std::vector<int> prefix_vec;

        if (this->is_character_based) {
            new_node = current_node->get_path_vec(prefix_vec, this->SPACE_ID, 1);
            current_node = new_node;
        } else {
            new_node = current_node->get_path_vec(prefix_vec, this->SPACE_ID);
            current_node = new_node->parent;  // Skipping spaces
        }

        // reconstruct word
        std::string word = vec2str(prefix_vec);
        ngram.push_back(word);
        no_start_token_count++;

        if (new_node->character == -1) {
            // No more spaces, but still need order
            for (int i = 0; i < this->window_length - order - 1; i++) {
                ngram.push_back(START_TOKEN);
            }
            break;
        }
    }
    std::reverse(ngram.begin(), ngram.end());
    std::pair<int, std::vector<std::string>> result(this->window_length - no_start_token_count, ngram);
    return result;
}

float HotWordsScorer::get_hotwords_score(const std::vector<std::string>& words, int offset) {
    float hotwords_score = 0;
    int words_size = words.size();
    std::unordered_map<std::string, float>::const_iterator iter;
    for (size_t index = 0; index < words_size; index++) {
        std::string word = "";
        if (this->is_character_based) {
            // contains at least two chinese characters.
            // words.end()-words.begin()-offset-index+1=words.size()-1-offset-index+1>=2
            if( words_size - offset - index <= 1) {
                break;
            }
            // chinese characters in fixed window, combining chinese characters into words.
            // word = std::accumulate(words.begin() + offset, words.end() - index, std::string{});
            word = std::accumulate(words.begin() + offset + index, words.end(), std::string{});
        } else {
            // word in fixed window, traverse each word in words, skip <s> token.
            if (index + offset >= words_size) {
                break;
            }
            word = words[index + offset];
        }
        iter = this->hotwords_dict.find(word);
        if (iter != this->hotwords_dict.end()) {
            hotwords_score += iter->second;
            // break loop after matching the hotwords.
            break;
        }
    }
    return hotwords_score;
}
