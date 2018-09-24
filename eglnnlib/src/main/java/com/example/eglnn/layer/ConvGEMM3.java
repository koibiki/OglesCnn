package com.example.eglnn.layer;

import android.content.Context;
import android.os.SystemClock;
import android.provider.Settings;
import android.text.TextUtils;

import com.example.eglnn.Render;
import com.example.eglnn.utils.ShaderUtils;
import com.example.eglnn.utils.TestDataCreator;
import com.example.eglnn.utils.Utils;

import java.nio.FloatBuffer;
import java.util.Locale;

import static com.example.eglnn.Render.initCompPro;
import static com.example.eglnn.utils.Constants.S_COMMON_SHADER_HEADER;
import static com.example.eglnn.utils.Constants.S_SUM_SHADER_HEADER;

/**
 *
 * */
public class ConvGEMM3 extends Layer {

    private static final String TAG = "ConvGEMM";
    private final int mKennelAmount;
    private PaddingType mPadType;

    private float[][][] mKennels;
    private int[] mStrides;
    private int[] mKennelShape;

    private int mConShaderPro;
    private int mSumShaderPro;

    private int mConNumGroupsX;
    private int mConNumGroupsY;
    private int mConvNumGroupsZ;

    private int[] mConvParams;
    private int[] mSumParams;

    private ActiveType mType;

    private String mKennelFilePath;

    private int mKennelTex;
    private int mConvOutTex;

    private int mPadW, mPadH;

    private int mSumNumGroupX;
    private int mSumNumGroupY;
    private int mSumNumGroupZ;


    private ConvGEMM3(Context context, Layer preLayer, int kAmount, int k_w, int k_h, int stride_w, int stride_h, ActiveType type, String kennelFilePath) {
        super(context, preLayer);
        this.mKennelShape = new int[]{k_w, k_h, preLayer.getOutputShape()[2]};
        this.mKennelAmount = kAmount;
        this.mStrides = new int[]{stride_w, stride_h};
        this.mType = type;
        this.mKennelFilePath = kennelFilePath;
    }

    public ConvGEMM3(Context context, Layer preLayer, int kAmount, int k_w, int k_h, PaddingType padType, int stride_w, int stride_h, ActiveType type, String kennelFilePath) {
        this(context, preLayer, kAmount, k_w, k_h, stride_w, stride_h, type, kennelFilePath);
        this.mPadType = padType;
        this.mOutShape = calculateConvShapeByType(kAmount);
    }

    public ConvGEMM3(Context context, Layer preLayer, int kAmount, int k_w, int k_h, int pad_w, int pad_h, int stride_w, int stride_h, ActiveType type, String kennelFilePath) {
        this(context, preLayer, kAmount, k_w, k_h, stride_w, stride_h, type, kennelFilePath);
        this.mPadW = pad_w;
        this.mPadH = pad_h;
        this.mOutShape = calculateConvShapeByPad(kAmount);
    }

    private int[] calculateConvShapeByType(int kennelAmount) {
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
        return new int[]{width, height, kennelAmount};
    }

    private int[] calculateConvShapeByPad(int kennelAmount) {
        int width = (int) Math.ceil((mInShape[0] + 2 * mPadW - mKennelShape[0]) * 1.0f / mStrides[0]);
        int height = (int) Math.ceil((mInShape[1] + 2 * mPadH - mKennelShape[1]) * 1.0f / mStrides[1]);
        return new int[]{width, height, kennelAmount};
    }

    private int getLocalSizeX(int[] outShape) {
        int outArea = outShape[0] * outShape[1];
        if (outArea >= 1024 * 4) {
            return 1024;
        } else {
            return (int) Math.ceil(outArea * 1.0f / 4);
        }
    }

    private int getLocalSizeY(int[] inShape, int xSize) {
        int maxSize = 1024 / xSize;
        float ySize = inShape[2] * 1.0f / 4;
        if (ySize >= maxSize) {
            return maxSize;
        } else {
            return (int) Math.floor(ySize);
        }
    }

    private int getLocalSizeZ(int xSize, int ySize) {
        int maxZSize = 1024 / xSize / ySize >= 64 ? 64 : 1024 / xSize / ySize;
        int zSize = Utils.alignBy4(mOutShape[2]) / 4;
        if (zSize >= maxZSize) {
            return maxZSize;
        } else {
            return zSize;
        }
    }

    private void initConv() {
        int xConSize = getLocalSizeX(mOutShape);
        mConNumGroupsX = (int) Math.ceil(Utils.alignBy4(mOutShape[0] * mOutShape[1]) * 1.0f / 4 / xConSize);

        int yConSize = getLocalSizeY(mInShape, xConSize);
        mConNumGroupsY = (int) Math.ceil(mInShape[2] * 1.0f / 4 / yConSize);

        int zConvSize = getLocalSizeZ(xConSize, yConSize);
        mConvNumGroupsZ = (int) Math.ceil(Utils.alignBy4(mOutShape[2]) * 1.0f / 4 / zConvSize);

        int xSumSize = mOutShape[0] >= 1024 ? 1024 : mOutShape[0];
        mSumNumGroupX = (int) Math.ceil(mOutShape[0] * 1.0f / xSumSize);

        int ySumSize = 1024 / xSumSize >= mOutShape[1] ? mOutShape[1] : 1024 / xSumSize;
        mSumNumGroupY = (int) Math.ceil(mOutShape[1] * 1.0f / ySumSize);

        int zSumSize = getLocalSizeZ(xSumSize, ySumSize);
        mSumNumGroupZ = (int) Math.ceil(Utils.alignBy4(mOutShape[2]) * 1.0f / 4 / zSumSize);

        String convSource = createShaderSource("conv_gemm3.comp", xConSize, yConSize, zConvSize);
        mConShaderPro = initCompPro(convSource);

        String sumSource = createsSumShaderSource("conv_gemm3_sum.comp", xSumSize, ySumSize, zSumSize);
        mSumShaderPro = initCompPro(sumSource);

        mAttachID = Layer.getDataAttachID();

        mConvOutTex = Render.createFloatTextureArray(mOutShape[0] * 2, mOutShape[1] * Utils.alignBy4(mInShape[2]) / 4 + 1, Utils.alignBy4(mOutShape[2]) * 2 / 4);
        mOutTex = Render.createFloatTextureArray(mOutShape[0] * 2, mOutShape[1] * Utils.alignBy4(mInShape[2]) / 4 + 1, Utils.alignBy4(mOutShape[2]) * 2 / 4);

        if (TextUtils.isEmpty(mKennelFilePath)) {
            mKennels = createTestKennels();
        } else {
            mKennels = loadKennels();
        }
        // kennel 最后一列为bias
        mKennelTex = Render.createKennelFloatTextureArray(mKennelShape[0] * mKennelShape[1] + 1, mKennelAmount, Utils.alignBy4(mInShape[2]) / 4);
        transferToKennelTex();

        createSumShaderParams();
        createConvShaderParams();
    }

    private void transferToKennelTex() {
        for (int a = 0; a < mKennelAmount; a++) {
            float[][] kennel = mKennels[a];
            for (int c = 0; c < kennel.length; c++) {
                Render.transferToTextureArrayFloat(FloatBuffer.wrap(kennel[c]), mKennelTex, 0, a, c, mKennelShape[0] * mKennelShape[1] + 1, 1, 1);
            }
        }
    }

    private String createShaderSource(String sourcePath, int xSize, int ySize, int zSize) {
        String source = ShaderUtils.loadFromAssetsFile(sourcePath, mContext.getResources());
        return String.format(Locale.getDefault(), S_COMMON_SHADER_HEADER, xSize, ySize, zSize) + source;
    }

    private String createsSumShaderSource(String sourcePath, int xSize, int ySize, int zSize) {
        String source = ShaderUtils.loadFromAssetsFile(sourcePath, mContext.getResources());
        return String.format(Locale.getDefault(), S_SUM_SHADER_HEADER,
                mKennelShape[0] * mKennelShape[1],
                mOutShape[0], mOutShape[1], mOutShape[2],
                mType.index,
                Utils.alignBy4(mInShape[2]) / 4,
                Utils.alignBy4(mOutShape[2]) / 4,
                xSize, ySize, zSize) + source;
    }

    private void createConvShaderParams() {
        mConvParams = new int[15];
        mConvParams[0] = mKennelShape[0];
        mConvParams[1] = mKennelShape[1];
        mConvParams[2] = mKennelShape[2];
        mConvParams[3] = mInShape[0];
        mConvParams[4] = mInShape[1];
        mConvParams[5] = mInShape[2];
        mConvParams[6] = mOutShape[0];
        mConvParams[7] = mOutShape[1];
        mConvParams[8] = mOutShape[2];
        mConvParams[9] = mStrides[0];
        mConvParams[10] = mStrides[1];
        mConvParams[11] = mType.index;
        mConvParams[12] = Utils.alignBy4(mInShape[2]);
        mConvParams[13] = -1 * mPadW;
        mConvParams[14] = -1 * mPadH;
    }

    private void createSumShaderParams() {
        mSumParams = new int[7];
        mSumParams[0] = mKennelShape[0] * mKennelShape[1];     // kennel_area
        mSumParams[1] = mOutShape[0];                          // out_shape
        mSumParams[2] = mOutShape[1];
        mSumParams[3] = mOutShape[2];
        mSumParams[4] = mType.index;                           // activate_type
        mSumParams[5] = Utils.alignBy4(mInShape[2]) / 4;       // in_max_depth
        mSumParams[6] = Utils.alignBy4(mOutShape[2]) / 4;      // out_max_depth
    }

    private float[][][] createTestKennels() {
        return TestDataCreator.createConvKennels(mKennelShape, mKennelAmount);
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
        Render.bindTextureArray(mOutTex, mAttachID, 0);
    }

    @Override
    protected void actualForwardProc(float[][] input) {
        Render.performConvolute(mConShaderPro, mConvParams, mPreLayer.getOutTex(), mConvOutTex, mKennelTex, mConNumGroupsX, mConNumGroupsY, mConvNumGroupsZ);
        Render.performConvolute(mSumShaderPro, mConvOutTex, mOutTex, mKennelTex, mSumNumGroupX, mSumNumGroupY, mSumNumGroupZ);
    }
}
