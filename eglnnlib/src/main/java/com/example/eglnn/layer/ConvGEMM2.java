package com.example.eglnn.layer;

import android.content.Context;
import android.text.TextUtils;

import com.example.eglnn.Render;
import com.example.eglnn.utils.DataUtils;
import com.example.eglnn.utils.ShaderUtils;
import com.example.eglnn.utils.Utils;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import static com.example.eglnn.Render.initCompPro;
import static com.example.eglnn.utils.Constants.S_CONV_GEMM_SHADER_HEADER;

/**
 *
 * */
public class ConvGEMM2 extends Layer {

    private static final String TAG = "ConvGEMM";
    private final int mKennelAmount;
    private PaddingType mPadType;

    private int mNumGroupsX;

    private float[][][] mKennels;
    private int[] mStrides;
    private int[] mKennelShape;
    private int mPad;
    private int mShaderPro;
    private int mNumGroupsZ;
    private int[] mParams;
    private int mType;
    private String mKennelFilePath;
    private int mKennelTex;

    public ConvGEMM2(Context context, Layer preLayer, int kAmount, int k_w, int k_h, int pad, PaddingType padType, int stride_w, int stride_h, int type, String kennelFilePath) {
        super(context, preLayer);
        this.mKennelShape = new int[]{k_w, k_h, preLayer.getOutputShape()[2]};
        this.mKennelAmount = kAmount;
        this.mPad = pad;
        this.mPadType = padType;
        this.mStrides = new int[]{stride_w, stride_h};
        this.mType = type;
        this.mOutShape = calculateConvShape(kAmount);
        this.mKennelFilePath = kennelFilePath;
    }

    private int[] calculateConvShape(int kennelAmount) {
        int[] inShape = mPreLayer.getOutputShape();
        int width = calculateLength(inShape[0], mKennelShape[0], mStrides[0]);
        int height = calculateLength(inShape[1], mKennelShape[1], mStrides[1]);
        return new int[]{width, height, kennelAmount};
    }

    private int calculateLength(int length, int kennelLen, int stride) {
        float out = (length + 2 * mPad - kennelLen) * 1.0f / stride;
        if (mPadType == PaddingType.SAME) {
            return (int) (Math.ceil(out) + 1);
        } else {
            return (int) (Math.floor(out) + 1);
        }
    }

    private int getLocalSizeX(int[] outShape) {
        int outArea = outShape[0] * outShape[1];
        if (outArea >= 1024 * 4) {
            return 1024;
        } else {
            return (int) Math.ceil(outArea * 1.0f / 4);
        }
    }

    private int getLocalSizeZ(int xSize) {
        int maxZSize = 1024 / xSize >= 64 ? 64 : 1024 / xSize;
        int zSize = Utils.alignBy4(mOutShape[2]) / 4;
        if (zSize >= maxZSize) {
            return maxZSize;
        } else {
            return zSize;
        }
    }

    private void initConv() {
        int xSize = getLocalSizeX(mOutShape);
        mNumGroupsX = (int) Math.ceil(Utils.alignBy4(mOutShape[0] * mOutShape[1]) * 1.0f / 4 / xSize);

        int zSize = getLocalSizeZ(xSize);
        mNumGroupsZ = (int) Math.ceil(Utils.alignBy4(mOutShape[2]) * 1.0f / 4 / zSize);

        String source = createShaderSource(xSize, zSize);
        mShaderPro = initCompPro(source);
        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createFloatTextureArray(mOutShape[0] + 1, mOutShape[1] + 1, Utils.alignBy4(mOutShape[2]) / 4);

        if (TextUtils.isEmpty(mKennelFilePath)) {
            mKennels = createTestKennels();
        } else {
            mKennels = loadKennels();
        }

        mKennelTex = Render.createKennelFloatTextureArray(mKennelShape[0] * mKennelShape[1] + 1, mKennelAmount, Utils.alignBy4(mInShape[2]) / 4);
        transferToKennelTex();

        createShaderParams();
    }

    private void transferToKennelTex() {
        for (int a = 0; a < mKennelAmount; a++) {
            float[][] kennel = mKennels[a];
            for (int c = 0; c < kennel.length; c++) {
                Render.transferToTextureArrayFloat(FloatBuffer.wrap(kennel[c]), mKennelTex, 0, a, c, mKennelShape[0] * mKennelShape[1] + 1, 1, 1);
            }
        }
    }

    private String createShaderSource(int xSize, int zSize) {
        String source = ShaderUtils.loadFromAssetsFile("conv_gemm2.comp", mContext.getResources());
        int kennelArea = mKennelShape[0] * mKennelShape[1];
        int kennelSize = kennelArea * Utils.alignBy4(mKennelShape[2]);
        return String.format(Locale.getDefault(), S_CONV_GEMM_SHADER_HEADER, kennelArea, mKennelAmount, kennelSize, xSize, 1, zSize) + source;
    }


    private void createShaderParams() {
        mParams = new int[14];
        mParams[0] = mKennelShape[0];
        mParams[1] = mKennelShape[1];
        mParams[2] = mKennelShape[2];
        mParams[3] = mInShape[0];
        mParams[4] = mInShape[1];
        mParams[5] = mInShape[2];
        mParams[6] = mOutShape[0];
        mParams[7] = mOutShape[1];
        mParams[8] = mOutShape[2];
        mParams[9] = mStrides[0];
        mParams[10] = mStrides[1];
        mParams[11] = mPad;
        mParams[12] = mType;
        mParams[13] = Utils.alignBy4(mInShape[2]);
    }

    private float[][][] createTestKennels() {
        return Render.createConvKennels(mKennelShape, mKennelAmount);
    }

    /**
     * TODO
     * channel需要4对齐, 依次排满一个通道后再排下一个通道的值, 最后4个值为 bias, 0, 0, 0
     */
    private float[][][] loadKennels() {
        return null;
    }


    @Override
    public void initialize() {
        initConv();
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureArray(mOutTex, mAttachID);
    }

    @Override
    protected void actualForwardProc(float[][] input) {
        Render.performConvoluteGEMM2(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex,mKennelTex, mNumGroupsX, mNumGroupsZ);
    }
}
