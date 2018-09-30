package com.example.eglnn.layer;

import android.content.Context;

import com.example.eglnn.Render;
import com.example.eglnn.utils.ShaderUtils;
import com.example.eglnn.utils.Utils;

import java.util.Locale;

import static com.example.eglnn.Render.initCompPro;
import static com.example.eglnn.utils.ComputeShaderUtils.getCompShaderLocalSizeY;
import static com.example.eglnn.utils.ComputeShaderUtils.getCompShaderLocalSizeZ;
import static com.example.eglnn.utils.Constants.S_POOLING_SHADER_HEADER;

/**
 * pool 层 以输出 shape 为基准生成计算器, 每个计算器只负责一个深度纹理上的一次pool计算
 * 默认pool 输出 w 小于 1024
 */
public class Pooling extends Layer {

    private int[] mKennelShape;
    private int[] mStrides;
    private int mShaderPro;
    private int[] mParams;
    private PaddingType mPadType;
    private int mPadW, mPadH;
    private int mNumGroupsX;
    private int mNumGroupsY;
    private int mNumGroupsZ;

    public Pooling(Context context, String name, Layer preLayer, int k_w, int k_h, PaddingType padType, int stride_w, int stride_h) {
        super(context, name, preLayer);
        this.mPadType = padType;
        this.mKennelShape = new int[]{k_w, k_h};
        this.mStrides = new int[]{stride_w, stride_h};
        this.mOutShape = calculateConvShapeByType();
    }

    private int[] calculateConvShapeByType() {
        int width, height;
        if (mPadType == PaddingType.SAME) {
            width = (int) Math.ceil(mInShape[0] * 1.0f / mStrides[0]);
            mPadW = ((width - 1) * mStrides[0] + mKennelShape[0] - mInShape[0]) / 2;
            height = (int) Math.ceil(mInShape[1] * 1.0f / mStrides[1]);
            mPadH = ((height - 1) * mStrides[1] + mKennelShape[1] - mInShape[1]) / 2;
        } else {
            width = (int) Math.ceil((mInShape[0] - mKennelShape[0] + 1) * 1.0f / mStrides[0]);
            height = (int) Math.ceil((mInShape[1] - mKennelShape[1] + 1) * 1.0f / mStrides[1]);
        }
        return new int[]{width, height, mInShape[2]};
    }

    private int getLocalSizeX(int[] outShape) {
        if (outShape[0] >= 1024) {
            return 1024;
        } else {
            return outShape[0];
        }
    }

    private int getLocalSizeY(int[] outShape, int xSize) {
        int maxSize = 1024 / xSize;
        if (outShape[1] <= maxSize) {
            return outShape[1];
        } else {
            return maxSize;
        }
    }

    private int getLocalSizeZ(int xSize, int ySize) {
        int maxZSize = 1024 / (xSize * ySize) >= 64 ? 64 : 1024 / (xSize * ySize);
        int zSize = Utils.alignBy4(mOutShape[2]) / 4;
        if (zSize >= maxZSize) {
            return maxZSize;
        } else {
            return zSize;
        }
    }

    private void initPooling() {
        int xSize = getLocalSizeX(mOutShape);
        mNumGroupsX = (int) Math.ceil(mOutShape[0] * 1.0f / xSize);

        int ySize = getLocalSizeY(mOutShape, xSize);
        mNumGroupsY = (int) Math.ceil(mOutShape[1] * 1.0f / ySize);

        int zSize = getLocalSizeZ(xSize, ySize);
        mNumGroupsZ = (int) Math.ceil(Utils.alignBy4(mOutShape[2]) * 1.0f / 4 / zSize);

        String source = createShaderSource(xSize, ySize, zSize);
        mShaderPro = initCompPro(source);
        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createFloatTextureArray(mOutShape[0], mOutShape[1], Utils.alignBy4(mOutShape[2]) / 4);

        mParams = new int[13];
        mParams[0] = mInShape[0];
        mParams[1] = mInShape[1];
        mParams[2] = mInShape[2];
        mParams[3] = mOutShape[0];
        mParams[4] = mOutShape[1];
        mParams[5] = mOutShape[2];
        mParams[6] = mKennelShape[0];
        mParams[7] = mKennelShape[1];
        mParams[8] = mStrides[0];
        mParams[9] = mStrides[1];
        mParams[10] = Utils.alignBy4(mInShape[2]) / 4;
        mParams[11] = -1 * mPadW;
        mParams[12] = -1 * mPadH;
    }

    private String createShaderSource(int xSize, int ySize, int zSize) {
        String shaderFile = "pooling.comp";
        String source = ShaderUtils.loadFromAssetsFile(shaderFile, mContext.getResources());
        int kennelArea = mKennelShape[0] * mKennelShape[1];
        return String.format(Locale.getDefault(), S_POOLING_SHADER_HEADER, kennelArea, xSize, ySize, zSize) + source;
    }

    @Override
    public void initialize() {
        initPooling();
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureArray(mOutTex, mAttachID);
    }

    @Override
    protected void actualForwardProc(float[][] input) {
        Render.performCompute(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mNumGroupsX, mNumGroupsY, mNumGroupsZ);
    }

}
