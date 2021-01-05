#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main()
{
    std::string arr[] = {"Hello", "C++", "World", "from", "VS Code", "and the C++ extension!"};
    std::vector<std::string> msg(arr, arr + sizeof(arr)/sizeof(std::string));

    for (const std::string& word : msg)
    {
        std::cout << word << " ";
    }
    std::cout << std::endl;
}
