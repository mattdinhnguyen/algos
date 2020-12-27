#include<stdio.h>
#include <string>

bool HasAnagramSubstring(const std::string& src, const std::string& target)
{
    if(target.size() > src.size()) return false;
    
    int srcLen = src.size(), targetLen = target.size();
    int targetCount[128] = {0}, count[128] = {0}, i, j; 
    //initialize
    for(i = 0; i < target.size(); ++i){
        ++targetCount[target[i]];
        ++count[src[i]];
    }
    //loop
    i = 0;
    while(true){
        //check if substring is an anagram
        for(j = 0; j < targetLen; ++j){
            if(count[target[j]] != targetCount[target[j]]) break;
        }
        if(j == targetLen) return true;
        //slide window
        if(i + 1 + targetLen > srcLen) break;
        --count[src[i]];
        ++count[src[i + targetLen]];
        ++i;
    }
    
    return false;
}

int main()
{
    printf("%s \n", HasAnagramSubstring("xyz", "afdgzyxksldfm") ? "true" : "false");
}
