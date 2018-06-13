package com.example.cnnlib.utils;

import java.util.Stack;

public class AttachIDManager {

    private static Stack<Integer> mStack;
    private static AttachIDManager mAttachIDManager;

    private AttachIDManager() {
        mStack = new Stack<>();
        for (int i = 31; i >= 0; i--) {
            mStack.push(i);
        }
    }

    public static AttachIDManager getInstance() {
        if (mAttachIDManager == null) {
            mAttachIDManager = new AttachIDManager();
        }
        return mAttachIDManager;
    }


    public int getAttachID() {
        return mStack.pop();
    }

    public void pushAttachID(int i) {
        mStack.push(i);
    }


}
