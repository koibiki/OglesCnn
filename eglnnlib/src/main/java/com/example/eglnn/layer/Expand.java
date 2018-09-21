package com.example.eglnn.layer;

import android.content.Context;
import android.text.TextUtils;

import com.example.eglnn.Render;
import com.example.eglnn.utils.ShaderUtils;
import com.example.eglnn.utils.TestDataCreator;
import com.example.eglnn.utils.Utils;

import java.nio.FloatBuffer;
import java.util.Locale;

import static com.example.eglnn.Render.initCompPro;
import static com.example.eglnn.utils.Constants.S_CONV_GEMM_SHADER_HEADER;

/**
 *
 * */
public class Expand extends Layer {

    private static final String TAG = "ConvGEMM";
    private int mKennelAmount;
    private int mNumGroupsX;

    private float[][][] mKennels1;
    private float[][][] mKennels2;

    private int mShaderPro;
    private int mNumGroupsZ;
    private int[] mParams;
    private ActiveType mType;
    private String mKennel1FilePath;
    private String mKennel2FilePath;
    private int mKennelTex1;
    private int mKennelTex2;
    private int[] mKennelShape1;
    private int[] mKennelShape2;


    public Expand(Context context, Layer preLayer, int kAmount, ActiveType type, String kennel1FilePath, String kennel2FilePath) {
        super(context, preLayer);
        this.mKennelShape1 = new int[]{1, 1, mInShape[2]};
        this.mKennelShape2 = new int[]{3, 3, mInShape[2]};
        this.mKennelAmount = kAmount;
        this.mType = type;
        this.mKennel1FilePath = kennel1FilePath;
        this.mKennel2FilePath = kennel2FilePath;
        this.mOutShape = new int[]{mInShape[0], mInShape[1], kAmount * 2};
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
        int zSize = Utils.alignBy4(mKennelAmount) / 4;
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
        mOutTex = Render.createFloatTextureArray(mOutShape[0], mOutShape[1], Utils.alignBy4(mOutShape[2]) / 4);

        if (TextUtils.isEmpty(mKennel1FilePath) | TextUtils.isEmpty(mKennel2FilePath)) {
            mKennels1 = createTestKennels(mKennelShape1);
            mKennels2 = createTestKennels(mKennelShape2);
        } else {
            mKennels1 = loadKennels();
            mKennels2 = loadKennels();
        }
        // kennel 最后一列为bias
        mKennelTex1 = Render.createKennelFloatTextureArray(mKennelShape2[0] * mKennelShape2[1] + 1, mKennelAmount, 2 * Utils.alignBy4(mKennelShape2[2]) / 4);

        transferToKennelTex(mKennels1, mKennelTex1, mKennelShape1,0);
        transferToKennelTex(mKennels2, mKennelTex1, mKennelShape2, Utils.alignBy4(mKennelShape1[2]) / 4);
        createShaderParams();
    }

    private void transferToKennelTex(float[][][] kennels, int kennelTex, int[] kennelShape, int zOffset) {
        for (int a = 0; a < kennels.length; a++) {
            float[][] kennel = kennels[a];
            for (int c = 0; c < kennel.length; c++) {
                Render.transferToTextureArrayFloat(FloatBuffer.wrap(kennel[c]), kennelTex, 0, a, c + zOffset, kennelShape[0] * kennelShape[1] + 1, 1, 1);
            }
        }
    }

    private String createShaderSource(int xSize, int zSize) {
        String shaderFile = "expand.comp";
        String source = ShaderUtils.loadFromAssetsFile(shaderFile, mContext.getResources());
        int kennelArea = mKennelShape1[0] * mKennelShape1[1];
        int kennelSize = kennelArea * Utils.alignBy4(mKennelShape1[2]);
        return String.format(Locale.getDefault(), S_CONV_GEMM_SHADER_HEADER, kennelArea, mKennelAmount, kennelSize, xSize, 1, zSize) + source;
    }


    private void createShaderParams() {
        mParams = new int[9];
        mParams[0] = mInShape[0];
        mParams[1] = mInShape[1];
        mParams[2] = mInShape[2];
        mParams[3] = mOutShape[0];
        mParams[4] = mOutShape[1];
        mParams[5] = mOutShape[2];
        mParams[6] = mType.index;
        mParams[7] = mKennelAmount / 4;
        mParams[8] = Utils.alignBy4(mInShape[2]) / 4;
    }

    private float[][][] createTestKennels(int[] kennelShape) {
        return TestDataCreator.createConvKennels(kennelShape, mKennelAmount);
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
        Render.performExpand(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mKennelTex1, mNumGroupsX, mNumGroupsZ);
    }
}
