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
import static com.example.eglnn.utils.Constants.S_EXPAND_SHADER_HEADER;

/**
 *
 * */
public class Expand extends Layer {

    private static final String TAG = "Expand";
    private int mKennelAmount;

    private float[][][] mKennels1;
    private float[][][] mKennels2;

    private int mShaderPro1;
    private int mShaderPro2;
    private int mNumGroupsX;
    private int mNumGroupsZ;
    private int[] mParams1;
    private int[] mParams2;
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
        mNumGroupsZ = (int) Math.ceil(Utils.alignBy4(mKennelAmount) * 1.0f / 4 / zSize);

        String source1 = createShaderSource(0, xSize, zSize);
        String source2 = createShaderSource(Utils.alignBy4(mKennelAmount) / 4, xSize, zSize);
        mShaderPro1 = initCompPro(source1);
        mShaderPro2 = initCompPro(source2);
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
        mKennelTex1 = Render.createKennelFloatTextureArray(mKennelShape1[0] * mKennelShape1[1] + 1, mKennelAmount, 2 * Utils.alignBy4(mKennelShape1[2]) / 4);
        mKennelTex2 = Render.createKennelFloatTextureArray(mKennelShape2[0] * mKennelShape2[1] + 1, mKennelAmount, 2 * Utils.alignBy4(mKennelShape2[2]) / 4);

        transferToKennelTex(mKennels1, mKennelTex1, mKennelShape1);
        transferToKennelTex(mKennels2, mKennelTex2, mKennelShape2);
        createShaderParams();
    }

    private void transferToKennelTex(float[][][] kennels, int kennelTex, int[] kennelShape) {
        for (int a = 0; a < kennels.length; a++) {
            float[][] kennel = kennels[a];
            for (int c = 0; c < kennel.length; c++) {
                Render.transferToTextureArrayFloat(FloatBuffer.wrap(kennel[c]), kennelTex, 0, a, c, kennelShape[0] * kennelShape[1] + 1, 1, 1);
            }
        }
    }

    private String createShaderSource(int startZ, int xSize, int zSize) {
        String shaderFile = "expand.comp";
        String source = ShaderUtils.loadFromAssetsFile(shaderFile, mContext.getResources());
        return String.format(Locale.getDefault(), S_EXPAND_SHADER_HEADER, startZ, xSize, 1, zSize) + source;
    }

    private void createShaderParams() {
        mParams1 = new int[12];
        mParams1[0] = mKennelShape1[0];
        mParams1[1] = mKennelShape1[1];
        mParams1[2] = mKennelShape1[2];
        mParams1[3] = mInShape[0];
        mParams1[4] = mInShape[1];
        mParams1[5] = mInShape[2];
        mParams1[6] = mOutShape[0];
        mParams1[7] = mOutShape[1];
        mParams1[8] = mOutShape[2] / 2;
        mParams1[9] = mType.index;
        mParams1[10] = Utils.alignBy4(mInShape[2]);
        mParams1[11] = 0;

        mParams2 = new int[12];
        mParams2[0] = mKennelShape2[0];
        mParams2[1] = mKennelShape2[1];
        mParams2[2] = mKennelShape2[2];
        mParams2[3] = mInShape[0];
        mParams2[4] = mInShape[1];
        mParams2[5] = mInShape[2];
        mParams2[6] = mOutShape[0];
        mParams2[7] = mOutShape[1];
        mParams2[8] = mOutShape[2] / 2;
        mParams2[9] = mType.index;
        mParams2[10] = Utils.alignBy4(mInShape[2]);
        mParams2[11] = -1;
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
        Render.bindTextureArray(mOutTex, mAttachID, mOutShape[2] / 4 - 1);
    }

    @Override
    protected void actualForwardProc(float[][] input) {
        Render.performExpand(mShaderPro1, mParams1, mPreLayer.getOutTex(), mOutTex, mKennelTex1, mNumGroupsX, mNumGroupsZ);
        Render.performExpand(mShaderPro2, mParams2, mPreLayer.getOutTex(), mOutTex, mKennelTex2, mNumGroupsX, mNumGroupsZ);
    }
}