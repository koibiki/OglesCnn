package com.example.eglnn.layer;

import android.content.Context;
import android.text.TextUtils;

import com.example.eglnn.Render;
import com.example.eglnn.utils.Numpy;
import com.example.eglnn.utils.ShaderUtils;
import com.example.eglnn.utils.TestDataCreator;
import com.example.eglnn.utils.Utils;

import java.nio.FloatBuffer;
import java.util.Locale;

import static com.example.eglnn.Render.initCompPro;
import static com.example.eglnn.utils.Constants.S_COMMON_SHADER_HEADER;

/**
 *
 * */
public class ConvWinogradF23 extends Layer {

    private static final String TAG = "ConvGEMM";
    private final int mKennelAmount;


    private float[][][][] mKennels;
    private int[] mStrides;
    private int[] mKennelShape;
    private int mShaderPro;
    private int mNumGroupsX;
    private int mNumGroupsY;
    private int mNumGroupsZ;
    private int[] mParams;
    private ActiveType mType;
    private String mKennelFilePath;
    private PaddingType mPadType;

    private int mIndexBufferId;
    private int mKennelTex;
    private int mPadW, mPadH;

    private ConvWinogradF23(Context context, Layer preLayer, int kAmount, int k_w, int k_h, int stride_w, int stride_h, ActiveType type, String kennelFilePath) {
        super(context, preLayer);
        this.mKennelShape = new int[]{k_w, k_h, preLayer.getOutputShape()[2]};
        this.mKennelAmount = kAmount;
        this.mStrides = new int[]{stride_w, stride_h};
        this.mType = type;
        this.mKennelFilePath = kennelFilePath;
    }

    public ConvWinogradF23(Context context, Layer preLayer, int kAmount, int k_w, int k_h, PaddingType padType, int stride_w, int stride_h, ActiveType type, String kennelFilePath) {
        this(context, preLayer, kAmount, k_w, k_h, stride_w, stride_h, type, kennelFilePath);
        this.mPadType = padType;
        this.mOutShape = calculateConvShapeByType(kAmount);
    }

    public ConvWinogradF23(Context context, Layer preLayer, int kAmount, int k_w, int k_h, int pad_w, int pad_h, int stride_w, int stride_h, ActiveType type, String kennelFilePath) {
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
        if (outShape[0] / 2 >= 1024) {
            return 1024;
        } else {
            return outShape[0] / 2;
        }
    }

    private int getLocalSizeY(int[] outShape, int xSize) {
        int maxSize = 1024 / xSize;
        if (outShape[1] / 2 <= maxSize) {
            return outShape[1] / 2;
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

    private void initConv() {
        int xSize = getLocalSizeX(mOutShape);
        mNumGroupsX = (int) Math.ceil(mOutShape[0] * 1.0f / 2 / xSize);

        int ySize = getLocalSizeY(mOutShape, xSize);
        mNumGroupsY = (int) Math.ceil(mOutShape[1] * 1.0f / 2 / ySize);

        int zSize = getLocalSizeZ(xSize, ySize);
        mNumGroupsZ = (int) Math.ceil(Utils.alignBy4(mOutShape[2]) * 1.0f / 4 / zSize);

        String source = createShaderSource(xSize, ySize, zSize);
        mShaderPro = initCompPro(source);
        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createFloatTextureArray(mOutShape[0], mOutShape[1], Utils.alignBy4(mOutShape[2]) * 2 / 4);

        if (TextUtils.isEmpty(mKennelFilePath)) {
            mKennels = createTestKennels();
        } else {
            mKennels = loadKennels();
        }

        mKennelTex = Render.createKennelFloatTextureArray(17, mKennelAmount, Utils.alignBy4(mKennelShape[2]) / 4);
        transferKennelToGgGt(mKennels);

        createShaderParams();
    }

    private void transferKennelToGgGt(float[][][][] mKennels) {
        int kennel_amount = mKennels.length;
        int kennel_channel = mKennels[0].length;

        float[][][][] GgGt = new float[kennel_amount][kennel_channel][4][4];
        float[][] G = new float[][]{{1, 0, 0}, {0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0, 0, 1}};
        float[][] Gt = Numpy.transpose(G);

        for (int a = 0; a < kennel_amount; a++) {
            for (int c = 0; c < kennel_channel; c++) {
                GgGt[a][c] = Numpy.dot(Numpy.dot(G, mKennels[a][c]), Gt);
            }
        }

        int align_c = Utils.alignBy4(kennel_channel);

        float[] biases = new float[kennel_amount];
        float[][][] GgGt_align = new float[kennel_amount][align_c / 4][16 * 4 + 4]; // depth == 0 处 最后一列第一个为 bias

        for (int a = 0; a < kennel_amount; a++) {
            for (int c = 0; c < kennel_channel; c++) {
                for (int h = 0; h < 4; h++) {
                    for (int w = 0; w < 4; w++) {
                        GgGt_align[a][c / 4][(w + h * 4) * 4 + c % 4] = GgGt[a][c][h][w];
                    }
                }
            }
        }
        for (int a = 0; a < kennel_amount; a++) {
            GgGt_align[a][0][16 * 4] = 0.01f * a;
        }

        // 传输到纹理
        for (int a = 0; a < kennel_amount; a++) {
            float[][] kennel = GgGt_align[a];
            int depth = kennel.length;
            for (int c = 0; c < depth; c++) {
                Render.transferToTextureArrayFloat(FloatBuffer.wrap(kennel[c]), mKennelTex, 0, a, c, 17, 1, 1);
            }
        }
    }

//
//    private void transferToKennelTex() {
//        for (int a = 0; a < mKennelAmount; a++) {
//            float[][] kennel = mKennels[a];
//            for (int c = 0; c < kennel.length; c++) {
//                Render.transferToTextureArrayFloat(FloatBuffer.wrap(kennel[c]), mKennelTex, 0, a, c, mKennelShape[0] * mKennelShape[1] + 1, 1, 1);
//            }
//        }
//    }

    private String createShaderSource(int xSize, int ySize, int zSize) {
        String source = ShaderUtils.loadFromAssetsFile("conv_winograd_2x3.comp", mContext.getResources());
        return String.format(Locale.getDefault(), S_COMMON_SHADER_HEADER, xSize, ySize, zSize) + source;
    }

    private void createShaderParams() {
        mParams = new int[11];
        mParams[0] = mInShape[0];
        mParams[1] = mInShape[1];
        mParams[2] = mInShape[2];
        mParams[3] = mOutShape[0];
        mParams[4] = mOutShape[1];
        mParams[5] = mOutShape[2];
        mParams[6] = mType.index;
        mParams[7] = Utils.alignBy4(mInShape[2]) / 4;
        mParams[8] = Utils.alignBy4(mOutShape[2]) / 4;
        mParams[9] = -1 * mPadW;
        mParams[10] = -1 * mPadH;
    }

    private float[][][][] createTestKennels() {
        return TestDataCreator.createConvKennels2(mKennelShape, mKennelAmount);
    }

    /**
     * TODO
     * channel需要4对齐, 依次排满一个通道后再排下一个通道的值, 最后4个值为 bias, 0, 0, 0
     */
    private float[][][][] loadKennels() {
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
        Render.performConvolute(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mKennelTex, mNumGroupsX, mNumGroupsY, mNumGroupsZ);
    }
}
