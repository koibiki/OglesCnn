package com.example.eglnn;

import android.opengl.GLES31;

import com.example.eglnn.utils.ShaderUtils;

import java.nio.Buffer;

import static android.opengl.GLES20.GL_CLAMP_TO_EDGE;
import static android.opengl.GLES20.GL_COLOR_ATTACHMENT0;
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
import static android.opengl.GLES20.glBindTexture;
import static android.opengl.GLES20.glBufferData;
import static android.opengl.GLES20.glBufferSubData;
import static android.opengl.GLES20.glFramebufferTexture2D;
import static android.opengl.GLES20.glGenBuffers;
import static android.opengl.GLES20.glGenTextures;
import static android.opengl.GLES20.glGetIntegerv;
import static android.opengl.GLES20.glGetUniformLocation;
import static android.opengl.GLES20.glLinkProgram;
import static android.opengl.GLES20.glReadPixels;
import static android.opengl.GLES20.glTexParameteri;
import static android.opengl.GLES20.glTexSubImage2D;
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
import static android.opengl.GLES30.glTexStorage3D;
import static android.opengl.GLES30.glTexSubImage3D;
import static android.opengl.GLES31.GL_COMPUTE_SHADER;
import static android.opengl.GLES31.GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS;
import static android.opengl.GLES31.GL_MAX_COMPUTE_WORK_GROUP_SIZE;
import static android.opengl.GLES31.GL_READ_ONLY;
import static android.opengl.GLES31.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT;
import static android.opengl.GLES31.GL_SHADER_STORAGE_BUFFER;
import static android.opengl.GLES31.GL_WRITE_ONLY;
import static android.opengl.GLES31.glBindImageTexture;
import static android.opengl.GLES31.glDispatchCompute;
import static android.opengl.GLES31.glMemoryBarrier;


public class Render {

    private static final String TAG = "Render";

    public static int getMaxDrawBuffers() {
        int[] maxBuffer = new int[1];
        glGetIntegerv(GL_MAX_DRAW_BUFFERS, maxBuffer, 0);
        return maxBuffer[0];
    }

    public static int getMaxWorkGroupSize() {
        int[] maxBuffer = new int[1];
        glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, maxBuffer, 0);
        return maxBuffer[0];
    }

    public static int initKernelBuffer(int size) {
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

    public static int createFloatTextureArray(int width, int height, int depth) {
        return createTextureArray(width, height, depth, GL_RGBA32F);
    }

    public static int createIntTextureArray(int width, int height, int depth) {
        return createTextureArray(width, height, depth, GL_RGBA32I);
    }

    private static int createTextureArray(int width, int height, int depth, int internalFormat) {
        int[] texture = new int[1];
        glGenTextures(1, texture, 0);
        glBindTexture(GL_TEXTURE_2D_ARRAY, texture[0]);
        glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, internalFormat, width, height, depth);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
        return texture[0];
    }

    public static void bindTextureAndBuffer(int texID, int attachID) {
        int attachment = GL_COLOR_ATTACHMENT0 + attachID;
        glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texID, 0);
    }

    public static void bindTextureAndBuffer(int texID, int attachID, int bufferId) {
        int attachment = GL_COLOR_ATTACHMENT0 + attachID;
        glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texID, 0);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferId);
    }

    public static void bindTextureArray(int texID, int attachID, int layer) {
        int attachment = GL_COLOR_ATTACHMENT0 + attachID;
        glFramebufferTextureLayer(GL_FRAMEBUFFER, attachment, texID, 0, layer);
    }

    public static void bindTextureArrayAndBuffer(int texID, int attachID, int layer, int bufferId) {
        int attachment = GL_COLOR_ATTACHMENT0 + attachID;
        glFramebufferTextureLayer(GL_FRAMEBUFFER, attachment, texID, 0, layer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferId);
    }

    public static void performConvolute(int compProg, int[] params, int inTex, int outTex, int kennelTex, int numGroupsX, int numGroupsY, int numGroupZ) {
        glUseProgram(compProg);
        glUniform1iv(glGetUniformLocation(compProg, "params"), params.length, params, 0);

        glBindImageTexture(0, inTex, 0, true, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(1, outTex, 0, true, 0, GL_WRITE_ONLY, GL_RGBA32F);
        glBindImageTexture(2, kennelTex, 0, true, 0, GL_READ_ONLY, GL_RGBA32F);

        glDispatchCompute(numGroupsX, numGroupsY, numGroupZ);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    public static void performExpand(int compProg, int[] params, int inTex, int outTex, int kennelTex, int numGroupsX, int numGroupZ) {
        glUseProgram(compProg);
        glUniform1iv(glGetUniformLocation(compProg, "params"), params.length, params, 0);

        glBindImageTexture(0, inTex, 0, true, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(1, outTex, 0, true, 0, GL_WRITE_ONLY, GL_RGBA32F);
        glBindImageTexture(2, kennelTex, 0, true, 0, GL_READ_ONLY, GL_RGBA32F);

        glDispatchCompute(numGroupsX, 1, numGroupZ);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    public static void performFullConnectSSBO(int compProg, int[] params, int inTex, int outTex, int buffer, int numGroupX) {
        glUseProgram(compProg);

        glUniform1iv(glGetUniformLocation(compProg, "params"), params.length, params, 0);

        glBindImageTexture(0, inTex, 0, true, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(1, outTex, 0, true, 0, GL_WRITE_ONLY, GL_RGBA32F);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);

        glDispatchCompute(numGroupX, 1, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    public static void performCompute(int compProg, int[] params, int inTex, int outTex, int numGroupsX, int numGroupsY, int numGroupZ) {
        glUseProgram(compProg);

        glUniform1iv(glGetUniformLocation(compProg, "params"), params.length, params, 0);

        glBindImageTexture(0, inTex, 0, true, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(1, outTex, 0, true, 0, GL_WRITE_ONLY, GL_RGBA32F);

        glDispatchCompute(numGroupsX, numGroupsY, numGroupZ);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    public static void performComputeWithoutPara(int compProg, int inTex, int outTex, int numGroupsX, int numGroupsY, int numGroupZ) {
        glUseProgram(compProg);

        glBindImageTexture(0, inTex, 0, true, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(1, outTex, 0, true, 0, GL_WRITE_ONLY, GL_RGBA32F);

        glDispatchCompute(numGroupsX, numGroupsY, numGroupZ);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    public static void performConcat(int compProg, int[] params, int inTex1, int inTex2, int outTex, int numGroupsX, int numGroupsY, int numGroupZ) {
        glUseProgram(compProg);

        glUniform1iv(glGetUniformLocation(compProg, "params"), params.length, params, 0);

        glBindImageTexture(0, inTex1, 0, true, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(1, inTex2, 0, true, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(2, outTex, 0, true, 0, GL_WRITE_ONLY, GL_RGBA32F);

        glDispatchCompute(numGroupsX, numGroupsY, numGroupZ);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    public static Buffer transferFromTexture(Buffer data, int attachID, int x, int y, int width, int height) {
        glReadBuffer(GL_COLOR_ATTACHMENT0 + attachID);
        glReadPixels(x, y, width, height, GL_RGBA, GL_FLOAT, data);
        return data;
    }

    public static void transferToTexture(Buffer data, int texID, int xOffset, int yOffset, int width, int height) {
        glBindTexture(GL_TEXTURE_2D, texID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, xOffset, yOffset, width, height, GL_RGBA, GL_FLOAT, data);
    }

    public static void transferToTextureArrayFloat(Buffer data, int texID, int xOffset, int yOffset, int zOffset, int width, int height, int depth) {
        glBindTexture(GL_TEXTURE_2D_ARRAY, texID);
        glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, xOffset, yOffset, zOffset, width, height, depth, GL_RGBA, GL_FLOAT, data);
    }

    public static void trainferToTexttureArrayInt(Buffer data, int texID, int xOffset, int yOffset, int zOffset, int width, int height) {
        glBindTexture(GL_TEXTURE_2D_ARRAY, texID);
        glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, xOffset, yOffset, zOffset, width, height, 1, GL_RGBA_INTEGER, GL_INT, data);
    }

    public static void transferToBuffer(Buffer data, int bufferId, int offset) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferId);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset * 4, data.capacity() * 4, data);
    }

}
