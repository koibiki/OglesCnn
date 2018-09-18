package com.example.eglnn;

import android.opengl.GLES20;
import android.opengl.GLES30;
import android.opengl.GLES31;

import com.example.eglnn.utils.ShaderUtils;
import com.example.eglnn.utils.Utils;

import java.nio.Buffer;

import static android.opengl.GLES20.GL_CLAMP_TO_EDGE;
import static android.opengl.GLES20.GL_FLOAT;
import static android.opengl.GLES20.GL_FRAMEBUFFER;
import static android.opengl.GLES20.GL_INT;
import static android.opengl.GLES20.GL_LINEAR;
import static android.opengl.GLES20.GL_RGBA;
import static android.opengl.GLES20.GL_TEXTURE_2D;
import static android.opengl.GLES20.GL_TEXTURE_MAG_FILTER;
import static android.opengl.GLES20.GL_TEXTURE_MIN_FILTER;
import static android.opengl.GLES20.GL_TEXTURE_WRAP_S;
import static android.opengl.GLES20.GL_TEXTURE_WRAP_T;
import static android.opengl.GLES20.glBindBuffer;
import static android.opengl.GLES20.glBufferData;
import static android.opengl.GLES20.glBufferSubData;
import static android.opengl.GLES20.glFramebufferTexture2D;
import static android.opengl.GLES20.glGenBuffers;
import static android.opengl.GLES20.glGetIntegerv;
import static android.opengl.GLES20.glGetUniformLocation;
import static android.opengl.GLES20.glLinkProgram;
import static android.opengl.GLES20.glReadPixels;
import static android.opengl.GLES20.glTexParameteri;
import static android.opengl.GLES20.glUniform1iv;
import static android.opengl.GLES20.glUseProgram;
import static android.opengl.GLES30.GL_DYNAMIC_COPY;
import static android.opengl.GLES30.GL_MAX_DRAW_BUFFERS;
import static android.opengl.GLES30.GL_RGBA32F;
import static android.opengl.GLES30.GL_RGBA32I;
import static android.opengl.GLES30.GL_RGBA_INTEGER;
import static android.opengl.GLES30.GL_TEXTURE_2D_ARRAY;
import static android.opengl.GLES30.glBindBufferBase;
import static android.opengl.GLES30.glFramebufferTextureLayer;
import static android.opengl.GLES30.glReadBuffer;
import static android.opengl.GLES31.GL_COMPUTE_SHADER;
import static android.opengl.GLES31.GL_READ_ONLY;
import static android.opengl.GLES31.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT;
import static android.opengl.GLES31.GL_SHADER_STORAGE_BUFFER;
import static android.opengl.GLES31.GL_WRITE_ONLY;
import static android.opengl.GLES31.glBindImageTexture;
import static android.opengl.GLES31.glDispatchCompute;
import static android.opengl.GLES31.glMemoryBarrier;
import static com.example.eglnn.utils.Constants.S_TEXTURE_SIZE;


public class Render {

    private static final String TAG = "Render";

    private static int sConTex = -1;

    public static int getMaxDrawBuffers() {
        int[] maxBuffer = new int[1];
        glGetIntegerv(GL_MAX_DRAW_BUFFERS, maxBuffer, 0);
        return maxBuffer[0];
    }

    public static int initKennelBuffer(int size) {
        int[] fFrame = new int[1];
        glGenBuffers(1, fFrame, 0);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, fFrame[0]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, size * 4, null, GL_DYNAMIC_COPY);
        return fFrame[0];
    }

    public static int initCompPro(String source) {
        int compProg = GLES31.glCreateProgram();
        ShaderUtils.vglAttachShaderSource(compProg, GL_COMPUTE_SHADER, source);
        glLinkProgram(compProg);
        return compProg;
    }

    public static int createTexture() {
        return createTexture(S_TEXTURE_SIZE, S_TEXTURE_SIZE, GL_RGBA32F);
    }

    public static int createTexture(int format) {
        return createTexture(S_TEXTURE_SIZE, S_TEXTURE_SIZE, format);
    }

    public static int createFloatTextureArray(int width, int height, int depth) {
        return createTextureArray(width, height, depth, GL_RGBA32F);
    }

    public static int createFloatTextureArray(int depth) {
        return createTextureArray(S_TEXTURE_SIZE, S_TEXTURE_SIZE, depth, GL_RGBA32F);
    }

    public static int createKennelFloatTextureArray(int width, int height, int depth) {
        return createTextureArray(width, height, depth, GL_RGBA32F);
    }

    public static int createIntTextureArray(int width, int height,int depth) {
        return createTextureArray(width, height, depth, GL_RGBA32I);
    }

    public static int getConvKennelTexture() {
        if (sConTex == -1) {
            sConTex = createTexture(8192, 8192, GL_RGBA32F);
        }
        return sConTex;
    }

    private static int createTexture(int width, int height, int format) {
        int[] texture = new int[1];
        GLES20.glGenTextures(1, texture, 0);
        GLES20.glBindTexture(GL_TEXTURE_2D, texture[0]);
        GLES30.glTexStorage2D(GL_TEXTURE_2D, 1, format, width, height);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        GLES31.glBindTexture(GL_TEXTURE_2D, 0);
        return texture[0];
    }

    private static int createTextureArray(int width, int height, int depth, int internalFormat) {
        int[] texture = new int[1];
        GLES20.glGenTextures(1, texture, 0);
        GLES20.glBindTexture(GL_TEXTURE_2D_ARRAY, texture[0]);
        GLES30.glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, internalFormat, width, height, depth);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        GLES31.glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
        return texture[0];
    }

    public static void bindTextureAndBuffer(int texID, int attachID) {
        int attachment = GLES20.GL_COLOR_ATTACHMENT0 + attachID;
        glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texID, 0);
    }

    public static void bindTextureAndBuffer(int texID, int attachID, int bufferId) {
        int attachment = GLES20.GL_COLOR_ATTACHMENT0 + attachID;
        glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texID, 0);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferId);
    }

    public static void bindTextureArray(int texID, int attachID) {
        int attachment = GLES31.GL_COLOR_ATTACHMENT0 + attachID;
        glFramebufferTextureLayer(GL_FRAMEBUFFER, attachment, texID, 0, 0);
    }

    public static void bindTextureArray(int texID, int attachID, int layer) {
        int attachment = GLES31.GL_COLOR_ATTACHMENT0 + attachID;
        glFramebufferTextureLayer(GL_FRAMEBUFFER, attachment, texID, 0, layer);
    }

    public static void bindBuffer(int bufferId) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferId);
    }

    public static void bindTextureArrayAndBuffer(int texID, int attachID, int bufferId) {
        int attachment = GLES31.GL_COLOR_ATTACHMENT0 + attachID;
        glFramebufferTextureLayer(GL_FRAMEBUFFER, attachment, texID, 0, 0);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferId);
    }

    public static void performConvoluteGEMM(int compProg, int[] params, int inTex, int outTex, int kennelTex, int buffer, int numGroupsX, int numGroupZ) {
        GLES20.glUseProgram(compProg);
        glUniform1iv(GLES20.glGetUniformLocation(compProg, "params"), params.length, params, 0);

        glBindImageTexture(0, inTex, 0, true, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(1, outTex, 0, true, 0, GL_WRITE_ONLY, GL_RGBA32F);
        glBindImageTexture(2, kennelTex, 0, true, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);

        glDispatchCompute(numGroupsX, 1, numGroupZ);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    public static void performConvoluteGEMM4(int compProg, int[] params, int inTex, int outTex, int convIndex, int kennelTex, int numGroupsX, int numGroupZ) {
        GLES20.glUseProgram(compProg);
        glUniform1iv(GLES20.glGetUniformLocation(compProg, "params"), params.length, params, 0);

        glBindImageTexture(0, inTex, 0, true, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(1, outTex, 0, true, 0, GL_WRITE_ONLY, GL_RGBA32F);
        glBindImageTexture(2, kennelTex, 0, true, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(3, convIndex, 0, true, 0, GL_READ_ONLY, GL_RGBA32I);

        glDispatchCompute(numGroupsX, 1, numGroupZ);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    public static void performConvoluteGEMM2(int compProg, int[] params, int inTex, int outTex, int kennelTex,int numGroupsX, int numGroupZ) {
        GLES20.glUseProgram(compProg);
        glUniform1iv(GLES20.glGetUniformLocation(compProg, "params"), params.length, params, 0);

        glBindImageTexture(0, inTex, 0, true, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(1, outTex, 0, true, 0, GL_WRITE_ONLY, GL_RGBA32F);
        glBindImageTexture(2, kennelTex, 0, true, 0, GL_READ_ONLY, GL_RGBA32F);

        glDispatchCompute(numGroupsX, 1, numGroupZ);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    public static void performConvoluteGEMM3(int compProg, int[] params, int inTex, int outTex,
                                             int buffer0, int buffer1, int buffer2, int buffer3,
                                             int numGroupsX, int numGroupZ) {
        GLES20.glUseProgram(compProg);
        glUniform1iv(GLES20.glGetUniformLocation(compProg, "params"), params.length, params, 0);

        glBindImageTexture(0, inTex, 0, true, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(1, outTex, 0, true, 0, GL_WRITE_ONLY, GL_RGBA32F);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffer1);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buffer2);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, buffer3);

        glDispatchCompute(numGroupsX, 1, numGroupZ);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    public static Buffer transferFromTexture(Buffer data, int attachID, int x, int y, int width, int height) {
        glReadBuffer(GLES31.GL_COLOR_ATTACHMENT0 + attachID);
        glReadPixels(x, y, width, height, GL_RGBA, GL_FLOAT, data);
        return data;
    }

    public static void transferToTexture(Buffer data, int texID, int xOffset, int yOffset, int width, int height) {
        GLES20.glBindTexture(GL_TEXTURE_2D, texID);
        GLES20.glTexSubImage2D(GL_TEXTURE_2D, 0, xOffset, yOffset, width, height, GL_RGBA, GL_FLOAT, data);
    }

    public static void transferToTextureArrayFloat(Buffer data, int texID, int xOffset, int yOffset, int zOffset, int width, int height, int depth) {
        GLES20.glBindTexture(GL_TEXTURE_2D_ARRAY, texID);
        GLES31.glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, xOffset, yOffset, zOffset, width, height, depth, GL_RGBA, GL_FLOAT, data);
    }

    public static void trainferToTexttureArrayInt(Buffer data, int texID, int xOffset, int yOffset, int zOffset, int width, int height) {
        GLES20.glBindTexture(GL_TEXTURE_2D_ARRAY, texID);
        GLES31.glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, xOffset, yOffset, zOffset, width, height, 1, GL_RGBA_INTEGER, GL_INT, data);
    }

    public static void transferToBuffer(Buffer data, int bufferId, int offset) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferId);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset * 4, data.capacity() * 4, data);
    }

    public static void performConvoluteTex(int compProg, int[] params, int inTex, int outTex, int buffer, int numGroupsY, int numGroupsZ) {
        glUseProgram(compProg);
        glUniform1iv(glGetUniformLocation(compProg, "params"), params.length, params, 0);

        glBindImageTexture(0, inTex, 0, false, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(1, outTex, 0, false, 0, GL_WRITE_ONLY, GL_RGBA32F);
        glBindImageTexture(2, buffer, 0, false, 0, GL_READ_ONLY, GL_RGBA32F);

        glDispatchCompute(1, numGroupsY, numGroupsZ);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    public static float[][][] createConvKennels(int[] mKennelShape, int mKennelAmount) {
        int k_w = mKennelShape[0];
        int k_h = mKennelShape[1];
        int k_c = mKennelShape[2];
        int align_c = Utils.alignBy4(k_c);
        float[][][] kennenls = new float[mKennelAmount][align_c / 4][(k_w * k_h + 1) * 4];
        for (int a = 0; a < mKennelAmount; a++) {
            for (int w = 0; w < k_w; w++) {
                for (int h = 0; h < k_h; h++) {
                    for (int c = 0; c < k_c; c++) {
                        kennenls[a][c / 4][(w + h * k_w) * 4 + c % 4] = 1;
                    }
                }
            }
            kennenls[a][0][k_w * k_h * 4] = 0.00001f * a;       // bias
        }
        return kennenls;
    }

}
