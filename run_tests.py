#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試執行腳本
提供方便的測試執行介面
"""
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """執行命令並顯示結果"""
    print("\n" + "=" * 80)
    print(f"🔧 {description}")
    print("=" * 80)
    print(f"執行命令: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description='運行測試套件')
    parser.add_argument(
        '--mode',
        choices=['all', 'unit', 'integration', 'security', 'api', 'fast', 'coverage'],
        default='all',
        help='測試模式'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='並行運行測試'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='詳細輸出'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='運行特定測試文件'
    )
    parser.add_argument(
        '--html',
        action='store_true',
        help='生成 HTML 報告'
    )
    
    args = parser.parse_args()
    
    # 基本命令
    cmd = ['pytest']
    
    # 詳細輸出
    if args.verbose:
        cmd.append('-v')
    
    # 並行測試
    if args.parallel:
        cmd.extend(['-n', 'auto'])
    
    # HTML 報告
    if args.html:
        cmd.extend(['--html=test_report.html', '--self-contained-html'])
    
    # 根據模式選擇測試
    if args.mode == 'unit':
        cmd.extend(['-m', 'unit'])
        description = '運行單元測試'
    elif args.mode == 'integration':
        cmd.extend(['-m', 'integration'])
        description = '運行整合測試'
    elif args.mode == 'security':
        cmd.extend(['-m', 'security'])
        description = '運行安全測試'
    elif args.mode == 'api':
        cmd.extend(['-m', 'api'])
        description = '運行 API 測試'
    elif args.mode == 'fast':
        cmd.extend(['-m', 'not slow'])
        description = '運行快速測試（跳過耗時測試）'
    elif args.mode == 'coverage':
        cmd.extend([
            '--cov=remote_server',
            '--cov-report=html',
            '--cov-report=term',
            '--cov-report=xml'
        ])
        description = '運行測試並生成覆蓋率報告'
    else:
        description = '運行所有測試'
    
    # 特定文件
    if args.file:
        cmd.append(args.file)
        description = f'運行 {args.file}'
    
    # 執行測試
    returncode = run_command(cmd, description)
    
    # 顯示結果
    print("\n" + "=" * 80)
    if returncode == 0:
        print("✅ 所有測試通過！")
    else:
        print("❌ 部分測試失敗")
    print("=" * 80)
    
    # 如果是覆蓋率模式，顯示報告位置
    if args.mode == 'coverage':
        print("\n📊 覆蓋率報告:")
        print("  - HTML: htmlcov/index.html")
        print("  - XML: coverage.xml")
        print("  - 終端機: 已顯示在上方")
    
    if args.html:
        print("\n📄 HTML 測試報告: test_report.html")
    
    return returncode


if __name__ == '__main__':
    sys.exit(main())

