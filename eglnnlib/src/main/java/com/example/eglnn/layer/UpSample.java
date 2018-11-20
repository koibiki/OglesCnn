package com.example.eglnn.layer;

import android.content.Context;

import com.example.eglnn.Render;
import com.example.eglnn.eglenv.GLES31BackEnv;
import com.example.eglnn.utils.DataUtils;
import com.example.eglnn.utils.ShaderUtils;
import com.example.eglnn.utils.Utils;

import java.nio.FloatBuffer;
import java.util.Locale;

import static com.example.eglnn.Render.initCompPro;
import static com.example.eglnn.utils.Constants.S_COMMON_SHADER_HEADER;


/**
 * 前接层 shape 为 （1，1，N）
 */
public class UpSample extends Layer {

    private FloatBuffer mOut;       // 每次只能读取一个深度上的数据
    private int mShaderPro;
    private int mBindLayer;         // 纹理绑定深度编号

    private int mXUpSample;
    private int mYUpSample;
    private int mNumGroupsX;
    private int mNumGroupsY;
    private int mNumGroupsZ;
    private int[] mParams;

    public UpSample(Context context, String name, Layer preLayer, int xUpSample, int yUpSample) {
        super(context, name, preLayer);
        this.mXUpSample = xUpSample;
        this.mYUpSample = yUpSample;
        this.mOutShape = new int[]{xUpSample * mInShape[0], yUpSample * mInShape[1], mInShape[2]};
        this.mOut = FloatBuffer.allocate(mOutShape[0] * mOutShape[1] * 4);
        this.mBindLayer = 0;
    }

    private int getLocalSizeX(int[] inShape) {
        if (inShape[0] >= GLES31BackEnv.getMaxWorkGroupSize()) {
            return GLES31BackEnv.getMaxWorkGroupSize();
        } else {
            return inShape[0];
        }
    }

    private int getLocalSizeY(int[] inShape, int xSize) {
        int maxSize = GLES31BackEnv.getMaxWorkGroupSize() / xSize;
        if (inShape[1] <= maxSize) {
            return inShape[1];
        } else {
            return maxSize;
        }
    }

    private int getLocalSizeZ(int[] inShape, int xSize, int ySize) {
        int maxZSize = GLES31BackEnv.getMaxWorkGroupSize() / (xSize * ySize) >= 64 ? 64 : GLES31BackEnv.getMaxWorkGroupSize() / (xSize * ySize);
        int zSize = Utils.alignBy4(inShape[2]) / 4;
        if (zSize >= maxZSize) {
            return maxZSize;
        } else {
            return zSize;
        }
    }

    @Override
    public void initialize() {
        int xSize = getLocalSizeX(mInShape);
        int ySize = getLocalSizeY(mInShape, xSize);
        int zSize = getLocalSizeZ(mInShape, xSize, ySize);

        mNumGroupsX = (int) Math.ceil(mOutShape[0] * 1.0f / xSize);
        mNumGroupsY = (int) Math.ceil(mOutShape[1] * 1.0f / ySize);
        mNumGroupsZ = (int) Math.ceil(Utils.alignBy4(mOutShape[2]) * 1.0f / 4 / zSize);

        String source = createShaderSource(xSize, ySize, zSize);
        mShaderPro = initCompPro(source);

        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createFloatTextureArray(mOutShape[0], mOutShape[1], Utils.alignBy4(mOutShape[2]) / 4);

        mParams = new int[2];
        mParams[0] = mXUpSample;
        mParams[1] = mYUpSample;
    }

    private String createShaderSource(int xSize, int ySize, int zSize) {
        String shaderFile = "up_sample.comp";
        String source = ShaderUtils.loadFromAssetsFile(shaderFile, mContext.getResources());
        return String.format(Locale.getDefault(), S_COMMON_SHADER_HEADER, xSize, ySize, zSize) + source;
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureArray(mOutTex, mAttachID, mBindLayer);
    }

    @Override
    protected void actualForwardProc(float[][] input) {
        Render.performCompute(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mNumGroupsX, mNumGroupsY, mNumGroupsZ);
    }

    @Override
    public float[] readResult() {
        DataUtils.readOutput(this, mOut, mOutShape[0], mOutShape[1]);
        return mOut.array();
    }

}
