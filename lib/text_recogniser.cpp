
#include "text_recogniser.hpp"


#include <iostream>
#include <iomanip>
#include <vector>
#include <tuple>
#include <vector>

#include "difflib/difflib.h"
#include "rapidfuzz/fuzz.hpp"
#include "rapidfuzz/utils.hpp"

namespace IDCardDetector {


    int min3(int a, int b, int c) {
        a = a < b ? a : b;
        return a < c ? a : c;
    }

    int LevenshteinDistance(std::string s, int s_len, std::string t, int t_len) {
        int cost;
        if (s_len == 0)return t_len;
        if (t_len == 0)return s_len;
        if (s[s_len - 1] == t[t_len - 1])cost = 0;
        else cost = 1;
        return min3(LevenshteinDistance(s, s_len - 1, t, t_len) + 1,
                    LevenshteinDistance(s, s_len, t, t_len - 1) + 1,
                    LevenshteinDistance(s, s_len - 1, t, t_len - 1) + cost);

    }

    int LevenshteinDP(std::string s, std::string t) {
        int dp[s.length() + 1][t.length() +
                               1];
        for (int i = 0; i <= s.length(); i++)
            dp[i][0] = i;
        for (int j = 1; j <= t.length(); j++)
            dp[0][j] = j;
        for (int j = 0; j < t.length(); j++) {
            for (int i = 0; i < s.length(); i++) {
                if (s[i] == t[j])dp[i + 1][j + 1] = dp[i][j];//No operation
                else dp[i + 1][j + 1] = min3(dp[i][j + 1] + 1, dp[i + 1][j] + 1, dp[i][j] + 1);
            }
        }
        return dp[s.length()][t.length()];
    }

    TextRecogniser::TextRecogniser(char const *_windowName) {
    }


    std::vector<std::string> split(std::string s, std::string delimiter) {
        size_t pos_start = 0, pos_end, delim_len = delimiter.length();
        std::string token;
        std::vector<std::string> res;

        while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
            token = s.substr(pos_start, pos_end - pos_start);
            pos_start = pos_end + delim_len;
            res.push_back(token);
        }

        res.push_back(s.substr(pos_start));
        return res;
    }

    int TextRecogniser::RecogniseName(std::string inputText, std::string name, std::string *recognisedValues) {
        std::string text = inputText;
        std::string delimiter = " ";
        std::vector<std::string> nameValues = split(name, delimiter);
        std::string resultText = "";
        for (auto namePart: nameValues) {
            std::string test = namePart;
            auto s = difflib::MakeSequenceMatcher<>(inputText, test);
            int i = 0;
            std::string tag;
            std::size_t i1, i2, j1, j2;
            if (s.get_opcodes().size() > 0) {
                std::tie(tag, i1, i2, j1, j2) = s.get_opcodes()[1];
                std::string foundText = inputText.substr(i1, i2 - i1);
                if (resultText.size() > 0) {
                    resultText = resultText.append(" ");
                }
                resultText = resultText.append(foundText);

            }
            *recognisedValues = resultText;
        }
        int distance = LevenshteinDP(name, *recognisedValues);
        return distance;
    }

    bool TextRecogniser::ProcessText(std::string inputText, std::string fullName, std::string date,
                                     std::map<std::string, int> *recognisedValues) {
        std::cout << std::endl;
        std::cout << "inputText: " << inputText << std::endl;
        std::cout << "textRecogniser: " << inputText << std::endl;
        std::cout << "fullName: " << fullName << std::endl;
        std::cout << "date: " << date << std::endl;
        std::string recognisedName;
        if (inputText.length() > 10) {
            int nameDistance = RecogniseName(inputText, fullName, &recognisedName);

            std::string recognisedDate;
            int recognisedDateDistance = RecogniseName(inputText, date, &recognisedDate);

            recognisedValues->insert({recognisedName, nameDistance});
            recognisedValues->insert({recognisedDate, recognisedDateDistance});
            std::cout << recognisedName << " " << nameDistance << " " << recognisedDate << " " << recognisedDateDistance
                      << " ";

        }
    }

}