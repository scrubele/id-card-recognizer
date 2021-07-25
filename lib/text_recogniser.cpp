
#include "text_recogniser.hpp"


#include <iostream>
#include <iomanip>
#include <vector>
#include <tuple>
#include <vector>

#include "difflib/difflib.h"

namespace IDCardDetector {


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

    bool TextRecogniser::RecogniseName(std::string inputText, std::string name, std::string *recognisedValues) {
        std::string text = inputText;
        std::string delimiter = " ";
        std::vector<std::string> nameValues = split(name, delimiter);
//        for (std::vector<std::string>::const_iterator i = nameValues.begin(); i != nameValues.end(); ++i)
//            std::cout << *i << ' ';
        for (auto namePart: nameValues) {
            std::string test = namePart;
            auto s = difflib::MakeSequenceMatcher<>(inputText, test);
            for (auto const &opcode : s.get_opcodes()) {
                std::string tag;
                std::size_t i1, i2, j1, j2;
                std::tie(tag, i1, i2, j1, j2) = opcode;
//                std::cout << inputText.substr(i1, i2 - i1) <<std::endl;
                std::cout << std::setw(7) << tag << " a[" << i1 << ":" << i2 << " (" << inputText.substr(i1, i2 - i1)
                          << ")" << " b[" << j1 << ":" << j2 << " (" << test.substr(j1, j2 - j1) << ")" << "\n";

            }
        }
        return false;
    }

    bool TextRecogniser::ProcessText(std::string inputText, std::string fullName, std::string date,
                                     std::string *recognisedValues) {
        std::cout << std::endl;
        std::cout << "textRecogniser: " << inputText << std::endl;
        std::cout << "fullName: " << fullName << std::endl;
        std::cout << "date: " << date << std::endl;
        std::string *recognisedName;
        RecogniseName(inputText, fullName, recognisedName);


    }

}