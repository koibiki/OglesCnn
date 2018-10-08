package com.example.eglnn.eglenv;

import android.opengl.GLES20;

import com.example.eglnn.Render;

public class GLES31BackEnv {

    private int mWidth;
    private int mHeight;
    private EGLHelper mEGLHelper;
    private int[] fFrame = new int[1];
    private static int mMaxWorkGroupSize;

    final static String TAG = "GLES31BackEnv";

    public GLES31BackEnv(final int width, final int height) {
        this.mWidth = width;
        this.mHeight = height;
        mEGLHelper = new EGLHelper();
        mEGLHelper.eglInit(width, height);
        initFBO();
        Render.getMaxDrawBuffers();
        mMaxWorkGroupSize = Render.getMaxWorkGroupSize();
    }

    public static int getMaxWorkGroupSize() {
        return mMaxWorkGroupSize;
    }

    private void initFBO() {
        GLES20.glGenFramebuffers(1, fFrame, 0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fFrame[0]);
    }

}
