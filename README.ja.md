[简体中文](README.md) | [한국어](README.ko.md) | [日本語](README.ja.md) | [English](README.en.md)

# HanziStyleForge Fusion

Windows 向けの実験的な漢字フォント再構築ツールです。`target.ttf` から書体スタイルを学習し、`ref.otf` から漢字構造を取得して、インストール可能な TTF フォントを生成します。

> 長時間の無人実行を想定し、チェックポイント再開、安全停止、自動再試行に対応しています。

## 主な機能

- `fonts/target.ttf` から全体および局所的な書体スタイルを学習します。
- `refs/ref.otf` のデフォルト字形が持つすべての漢字を再構築します。
- 中国大陸、台湾、香港、日本、韓国などの参照フォントを利用できます。
- 対象フォントのラテン文字、数字、記号、仮名、ハングル、主要な OpenType データを可能な限り保持します。
- 学習、生成、候補選択、QA、ベクトル化、フォント構築を自動化します。

## 処理の概要

```text
target.ttf：スタイル
        +
ref.otf：漢字構造と対象範囲
        ↓
Style Encoder → VQ → Diffusion → Refiner / Retrieval / IDS
        ↓
候補選択 → QA → 輪郭変換 → TTF
```

プログラムは地域字形の正誤を判断しません。最終的な漢字構造は `ref.otf` のデフォルト Unicode `cmap` 字形に従います。

## 動作環境

- Windows 11 64-bit
- CUDA 対応 NVIDIA GPU
- Python 3.10 以降
- 150 GB 以上の空き容量を推奨

入力フォント：

```text
fonts\target.ttf
refs\ref.otf
```

静的フォントを推奨します。`target.ttf` には TrueType `glyf` テーブルが必要です。`ref.otf` は静的 TrueType または静的 CFF OTF を使用できます。可変フォント、TTC、OTC は使用しないでください。

## クイックスタート

1. リポジトリをダウンロードまたはクローンします。
2. スタイル元フォントを `fonts\target.ttf` に配置します。
3. 構造参照フォントを `refs\ref.otf` に配置します。
4. 環境をインストールします。

   ```text
   install_cuda130.bat
   ```

5. プロジェクトを確認します。

   ```text
   verify_project.bat
   ```

6. 完全な処理を開始または再開します。

   ```text
   run_months_resilient.bat
   ```

7. 安全停止：

   ```text
   request_safe_stop.bat
   ```

8. 再開前に停止マーカーを削除します。

   ```text
   clear_safe_stop.bat
   ```

## 出力

主な出力：

```text
build\target-HanziStyleForge-Fusion.ttf
build\target-HanziStyleForge-Fusion.ttf.report.json
work_hanzistyleforge_fusion_months\qa\index.html
```

学習データ、チェックポイント、生成進捗は次の場所に保存されます。

```text
work_hanzistyleforge_fusion_months\
```

学習中はこのフォルダーを削除しないでください。

## 使用前の注意

- 完全な処理には数日、数週間、またはそれ以上かかる場合があります。
- リポジトリにはフォント、事前学習済み重み、第三者フォントデータセットは含まれません。
- 生成フォントには `target.ttf` と `ref.otf` の両方のライセンスが適用される場合があります。
- 学習、変更、再配布の権利を持つフォントだけを使用してください。
- 本プロジェクトは実験的です。公開前に QA ページと最終フォントを確認してください。

## 研究・参考資料

HanziStyleForge Fusion は独立実装です。以下のプロジェクトと論文はアーキテクチャ設計の参考です。上流のソースコード、事前学習済み重み、フォントデータセットは本リポジトリに同梱されていません。

| 出典 | 参考にした方向 |
|---|---|
| [zi2zi](https://github.com/kaonashi-tyc/zi2zi) | 漢字スタイル変換、内容とスタイルの分離 |
| [FontDiffuser](https://github.com/yeungchenwa/FontDiffuser) | 拡散生成、マルチスケール内容集約、明示的スタイル制約 |
| [HanziGen](https://github.com/wangwenho/HanziGen) | VQ 表現と条件付き潜在拡散 |
| [VQ-Font](https://github.com/Yaomingshuai/VQ-Font) | 離散フォント token と構造認識強化 |
| [LF-Font / MX-Font](https://github.com/clovaai/fewshot-font-generation) | 局所部品スタイル、因子分解、複数専門家 |
| [DeepVecFont-v2](https://github.com/yizhiwang96/deepvecfont-v2) | Transformer ベクトル系列と輪郭補正 |
| [Efficient and Scalable Chinese Vector Font Generation via Component Composition](https://arxiv.org/abs/2404.06779) | 部品領域変換と大規模合成 |
| [cjkvi/cjkvi-ids](https://github.com/cjkvi/cjkvi-ids) | Unicode IDS 部品構造と局所領域ヒント |

引用は手法上の参考を示すだけであり、上流のコード、重み、データ、フォントをコピーする許可ではありません。第三者資料を使用する前に、現在のライセンスと利用条件を確認してください。

[zi2zi-JiT](https://github.com/kaonashi-tyc/zi2zi-JiT) は下に別項として記載します。アーキテクチャの参考にとどまらず、任意選択の生成バックエンドとして利用できるためです。

## 任意選択の生成バックエンド: zi2zi-JiT

生成段階はプラグイン式です。既定のバックエンドは本プロジェクト独自の Style Encoder → VQ → Diffusion → Refiner です。その代わりに、生成を [zi2zi-JiT](https://github.com/kaonashi-tyc/zi2zi-JiT)（事前学習済み重みを提供するピクセル空間拡散 Transformer）へ委譲できます。その場合も候補選別、IDS 部品検証、QA、精緻化、輪郭変換、TTF 構築という下流工程はすべて HanziStyleForge Fusion が担当します。

zi2zi-JiT のソースコードも重みも本リポジトリには同梱していません。上流リポジトリのクローンと重みのダウンロードはご自身で行い、バックエンドはそのローカルコピーを呼び出します。

### 使い方

バックエンドは `config.json` の `backend` ブロックで選択し、`--backend` で一時的に上書きできます。

```text
hanzistyleforge.py --backend=zi2zi-jit fusion-generate
```

指定できる値は `native`（既定。本プロジェクト独自の生成スタック）、`zi2zi-jit`、`dir` です。`dir` は生成済み画像のディレクトリを読み込みます。手動で生成した結果を取り込む場合や、生成器に依存せず後処理工程だけを検証する場合に使います。

```json
"backend": {
  "name": "zi2zi-jit",
  "candidate_count": 3,
  "zi2zi_jit": {
    "repo_dir": "D:/zi2zi-JiT",
    "checkpoint": "D:/zi2zi-JiT/run/lora_target/checkpoint-last.pth",
    "font_label": 0
  }
}
```

`python_executable` を空にすると HanziStyleForge を実行しているインタプリタを再利用します。zi2zi-JiT の推論経路に必要なのは torch、numpy、opencv、einops だけで、`environment.yaml` で固定された一式は不要です。

### 先に LoRA ファインチューニングが必要です

**公開されている JiT-B/16 の重みは事前学習の成果物であり、ゼロショットでは使えません。** 未知のフォントにそのまま適用すると画が系統的に欠落します。zi2zi-JiT の README にある生成例はすべてファインチューニング済みの重みを使っています。

`scripts/generate_font_dataset.py` でデータセットを作成します。ソースフォントには推論時に与えるものと同じ `ref.otf` を使ってください。事前学習に合わせるより推論時の内容分布に合わせるほうが重要です。続いて `lora_single_gpu_finetune_jit.py` を実行し、得られた重みを `checkpoint` に指定して `font_label` を `0` にします（単一フォントのデータセットは `001_<name>` として配置されるため）。`font_label` を未設定にすると label-drop トークンを使いますが、それが意味を持つのはベース重みの場合だけです。

Windows では加えて `TORCHDYNAMO_DISABLE=1`（Triton の Windows ビルドが存在しないため）、リポジトリ直下を指す `PYTHONPATH`（`scripts/` 配下のスクリプトは自身のディレクトリが `sys.path[0]` になるため）、`--num_workers 0`（DataLoader のワーカーが lambda を含む dataset を pickle する必要があるため）、`--online_eval` を付けないこと（FID を計算するが PyPI の torch-fidelity は上流が使う fork と API が異なるため）が必要です。

### バックエンドに別のトポロジー閾値を設ける理由

全体の `topology` 閾値は組み込み生成器向けに較正されています。組み込み生成器は参照に structure-lock されるため、その骨格に非常に近い結果を出します。真のスタイル変換を行うバックエンドは設計上そこから乖離するため、同じ閾値ではすべての出力が却下されます（実測で `topology_score` の中央値は 0.14、上限は 0.06）。

そこで `backend.topology` は非 native バックエンドに対して骨格類似度の上限だけを緩めます。**緩めないのは連結成分・穴・オイラー数の差**で、これらはゼロのまま保たれ、生成された字が同じ文字であることを保証します。同じ実測でこれらの中央値はすでにゼロだったため、正常なスタイル変換は通過し、画が増減した字は却下されて参照側にフォールバックします。

`selection.csv` の信頼度も同じ基準で較正されます。これは候補が基準の内側にどれだけ余裕を持っているかを表すため、緩和されたバックエンド経路の値は native 経路より構造的に低くなります。そのため QA が低信頼度と判定する閾値 `qa.low_confidence_threshold` を設定可能にしました（既定値 0.75 は native の較正に対応）。600 字の実測ではバックエンドの分布は p10=0.125、p50=0.258、p90=0.486 で、既定値ではすべての字が低信頼度として報告されます。`config_zi2zi_production.json` では 0.12 を指定し、最も悪い約 1 割だけを対象にしています。

同じ理由で、**非 native バックエンドでは `refine` 段階をスキップします。** 長時間精錬は参照構造に最も近い候補を探索し、参照フォールバックとの混合によってそこへ到達します。組み込み生成器のノイズを含む出力にとっては純化ですが、スタイル変換に対しては消去であり、実測では 40 字のうち 32 字が参照字形に置き換えられました。`build` は `refined/selection.csv` を優先するため、有効なままでは参照輪郭でほぼ構成されたフォントが出来上がります。強制したい場合は `backend.run_refine=true` を設定してください。

> **表示義務。** zi2zi-JiT のコードは MIT ライセンスですが、「Font Artifact License Addendum」が生成物に追加条件を課します。その出力から作られた文字が **200 文字を超える** フォント製品を配布する場合、出典表示が必要です。本ツールの通常の実行では 200 文字をはるかに超えるため、このバックエンドを使ったなら表示が必要だと考えてください。"Created using zi2zi-JiT artifacts" と記載し、上流リポジトリへのリンクを添えます。既定のバックエンドで生成したフォントには適用されません。詳細は `THIRD_PARTY_NOTICES.md` を参照してください。

## コントリビューション

Issue と Pull Request を歓迎します。第三者のコード、データ、モデルを追加する場合は、出典とライセンス情報を明記してください。
