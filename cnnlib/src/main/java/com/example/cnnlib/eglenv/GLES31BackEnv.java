package com.example.cnnlib.eglenv;

import android.util.Log;

import com.example.cnnlib.render.ComputeRender;

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

    String mThreadOwner;

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

    public void setThreadOwner(String threadOwner) {
        this.mThreadOwner = threadOwner;
    }

//    public void performConvolute(ComputeRender render) {
//        if (render == null) {
//            Log.e(TAG, " Renderer was not set.");
//            return;
//        }
//        if (!Thread.currentThread().getName().equals(mThreadOwner)) {
//            Log.e(TAG, "getBitmap: This thread does not own the OpenGL context.");
//            return;
//        }
//        render.performConvolute();
//    }

    public void destroy() {
        mEGLHelper.destroy();
    }


}
