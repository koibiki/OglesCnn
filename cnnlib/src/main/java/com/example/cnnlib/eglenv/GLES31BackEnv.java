package com.example.cnnlib.eglenv;

import static android.opengl.GLES31.GL_FRAMEBUFFER;
import static android.opengl.GLES31.glBindFramebuffer;
import static android.opengl.GLES31.glGenFramebuffers;

/**
 * Description:
 */
public class GLES31BackEnv {

    private int mWidth;
    private int mHeight;
    private EGLHelper mEGLHelper;

    private int[] fFrame = new int[1];

    final static String TAG = "GLES31BackEnv";

    public GLES31BackEnv(int width, int height) {
        this.mWidth = width;
        this.mHeight = height;
        this.mEGLHelper = new EGLHelper();
        this.mEGLHelper.eglInit(width, height);
        initFBO();
    }

    private void initFBO() {
        glGenFramebuffers(1, fFrame, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, fFrame[0]);
    }

    public void destroy() {
        mEGLHelper.destroy();
    }


}
