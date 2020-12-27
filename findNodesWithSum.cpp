#include <iostream>

using namespace std;

// Define BST and stacks (using linked-list)

class BSTNode {
public:
    int value;
    BSTNode *left;
    BSTNode *right;

    BSTNode(int value, BSTNode *left=NULL, BSTNode *right=NULL) {
        this->value = value;
        this->left = left;
        this->right = right;
    }
    int contains(int value, BSTNode *exclude=NULL) {
        if (value < this->value)
            return (this->left != NULL && this->left->contains(value, exclude));
        else if (value > this->value)
            return (this->right != NULL && this->right->contains(value, exclude));
        else
            return (this == exclude) ? 0 : 1;
    }
};

class LLNode {
public:
    BSTNode *tree;
    LLNode *next;
    LLNode(BSTNode *tree, LLNode *next) {
        this->tree = tree;
        this->next = next;
    }
};

class Stack {
public:
    LLNode *head;
    Stack() {
        this->head = NULL;
    }
    ~Stack() {
        LLNode *oldHead;
        while (this->head != NULL) {
            oldHead = this->head;
            this->head = oldHead->next;
            delete oldHead;
        }
    }
    void push(BSTNode *tree) {
        this->head = new LLNode(tree, this->head);
    }
    BSTNode *pop() {
        LLNode *oldHead = this->head;
        if (oldHead == NULL)
            return NULL;
        BSTNode *ret = oldHead->tree;
        this->head = oldHead->next;
        delete oldHead;
        return ret;
    }
};

/*
    We performe a depth-first search on the BST using a stack which size won't
    grow over O(height of the tree).
    For each node we find, we check in logarithmic time whether the complement is
    in the tree (taking care of excluding this node).
*/
int findNodesWithSum(BSTNode *root, int sum) {
    Stack stack;
    stack.push(root);
    BSTNode *curTree;
    while ((curTree = stack.pop()) != NULL) {
        if (root->contains(sum - curTree->value, curTree)) {
            cout << curTree->value << "+" << (sum - curTree->value) << endl;
            return 1;
        }
        if (curTree->right != NULL)
            stack.push(curTree->right);
        if (curTree->left != NULL)
            stack.push(curTree->left);
    }
    return 0;
}

int main(int argc, char* argv[]) {
    /*
                    13
            5               15
        3       8       14      17
    */
    BSTNode *root = new BSTNode(13,
                        new BSTNode(5,
                            new BSTNode(3),
                            new BSTNode(8)),
                        new BSTNode(15,
                            new BSTNode(14),
                            new BSTNode(17))
                        );
    // findNodesWithSum(root, 10);     // Not found
    // findNodesWithSum(root, 9);      // 5+8
    findNodesWithSum(root, 13);     // Not found
    findNodesWithSum(root, 20);     // 5+15
    return 0;
}
