#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全檢測腳本 (優化版)
執行靜態分析 (bandit) 和依賴漏洞掃描 (pip-audit)
只執行一次掃描,生成統一的 HTML 報告
"""

import subprocess  # nosec B404 B603 - subprocess 用於 FFmpeg 音訊轉換，參數已驗證
import sys
import os
import json
from datetime import datetime
from pathlib import Path

def print_section(title):
    """列印區段標題"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def run_command(command, description):
    """執行命令並返回結果"""
    print(f"\n⏳ 正在執行: {description}...")

    try:
        # 執行安全工具(bandit, pip-audit),命令參數由腳本控制
        result = subprocess.run(  # nosec B603
            command,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        return result
    except Exception as e:
        print(f"❌ 執行失敗: {str(e)}")
        return None

def check_tool_installed(tool_name, install_command):
    """檢查工具是否已安裝"""
    # 執行 pip show 檢查工具安裝狀態,工具名由腳本指定
    result = subprocess.run(  # nosec B603
        [sys.executable, "-m", "pip", "show", tool_name],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"⚠️  {tool_name} 尚未安裝")
        print(f"請執行: {install_command}")
        return False
    return True

def run_bandit():
    """執行 bandit 靜態分析 (僅一次)"""
    print_section("Bandit 靜態程式碼分析")

    if not check_tool_installed("bandit", "pip install bandit"):
        return False, None

    # 創建 security_reports 目錄
    os.makedirs("security_reports", exist_ok=True)

    # 執行 bandit (僅 JSON 格式,只執行一次)
    result = run_command(
        [sys.executable, "-m", "bandit", "-r", "backend", "frontend/src",
         "-f", "json",
         "--exclude", "./.git,./node_modules,./venv,./venv_*,./.venv"],
        "Bandit 靜態分析"
    )

    if not result:
        return False, None

    # 解析 JSON 結果
    try:
        # Bandit 直接輸出到 stdout
        data = json.loads(result.stdout) if result.stdout else {}

        # 解析結果
        results = data.get('results', [])
        metrics = data.get('metrics', {})

        # 統計總數
        total_issues = len(results)
        high_severity = sum(1 for r in results if r.get('issue_severity') == 'HIGH')
        medium_severity = sum(1 for r in results if r.get('issue_severity') == 'MEDIUM')
        low_severity = sum(1 for r in results if r.get('issue_severity') == 'LOW')

        print(f"\n📊 掃描結果總覽:")
        print(f"   總問題數: {total_issues}")
        print(f"   🔴 高風險: {high_severity}")
        print(f"   🟡 中風險: {medium_severity}")
        print(f"   🟢 低風險: {low_severity}")

        # 顯示高/中風險問題
        if results:
            critical_issues = [r for r in results if r.get('issue_severity') in ['HIGH', 'MEDIUM']]
            if critical_issues:
                print(f"\n⚠️  發現的重要安全問題:")
                print("-" * 80)

                # 按嚴重程度排序
                severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
                sorted_results = sorted(critical_issues,
                                      key=lambda x: severity_order.get(x.get('issue_severity', 'LOW'), 3))

                for i, issue in enumerate(sorted_results[:15], 1):  # 顯示前15個
                    severity = issue.get('issue_severity', 'UNKNOWN')
                    confidence = issue.get('issue_confidence', 'UNKNOWN')
                    test_id = issue.get('test_id', '')
                    filename = issue.get('filename', '').replace('\\', '/')
                    line_number = issue.get('line_number', '')
                    issue_text = issue.get('issue_text', '')

                    # 嚴重程度圖標
                    severity_icon = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(severity, '⚪')

                    print(f"\n{i}. {severity_icon} [{severity}/{confidence}] {test_id}")
                    print(f"   位置: {filename}:{line_number}")
                    print(f"   描述: {issue_text}")

                if len(sorted_results) > 15:
                    print(f"\n... 還有 {len(sorted_results) - 15} 個問題 (查看完整報告)")
        else:
            print("\n✅ 未發現安全問題!")

        print(f"\n✅ Bandit 掃描完成")
        return True, data

    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失敗: {str(e)}")
        print(f"輸出內容前500字: {result.stdout[:500]}")
        return False, None
    except Exception as e:
        print(f"❌ 處理失敗: {str(e)}")
        return False, None

def run_pip_audit():
    """執行 pip-audit 依賴漏洞掃描 (僅一次)"""
    print_section("pip-audit 依賴漏洞掃描")

    if not check_tool_installed("pip-audit", "pip install pip-audit"):
        return False, None

    # 執行 pip-audit (僅 JSON 格式,只執行一次)
    result = run_command(
        [sys.executable, "-m", "pip_audit", "--format", "json"],
        "pip-audit 漏洞掃描"
    )

    if not result:
        return False, None

    # 解析 JSON 結果
    try:
        # pip-audit 直接輸出到 stdout
        data = json.loads(result.stdout) if result.stdout else {"dependencies": []}

        dependencies = data.get('dependencies', [])
        # 只統計有漏洞的套件
        vulnerable_packages = [dep for dep in dependencies if dep.get('vulns', [])]
        total_vulnerabilities = sum(len(dep.get('vulns', [])) for dep in vulnerable_packages)

        print(f"\n📊 掃描結果總覽:")
        print(f"   掃描套件總數: {len(dependencies)}")
        print(f"   🚨 有漏洞的套件: {len(vulnerable_packages)}")
        print(f"   🚨 總漏洞數: {total_vulnerabilities}")

        if vulnerable_packages:
            print(f"\n🔍 發現的漏洞詳情:")
            print("-" * 80)

            for pkg_index, dep in enumerate(vulnerable_packages, 1):
                name = dep.get('name', '')
                version = dep.get('version', '')
                vulns = dep.get('vulns', [])

                print(f"\n📦 套件 {pkg_index}: {name} (版本 {version})")
                print(f"   發現 {len(vulns)} 個漏洞:")

                for vuln_index, vuln in enumerate(vulns, 1):
                    vuln_id = vuln.get('id', '')
                    fix_versions = vuln.get('fix_versions', [])
                    aliases = vuln.get('aliases', [])

                    print(f"\n   {vuln_index}. 🚨 {vuln_id}")
                    if aliases:
                        print(f"      別名: {', '.join(aliases)}")
                    if fix_versions:
                        print(f"      ✅ 修復版本: {', '.join(fix_versions)}")
                    else:
                        print(f"      ⚠️  尚無修復版本")

            print(f"\n{'-' * 80}")
            print("\n💡 修復建議:")
            for dep in vulnerable_packages:
                name = dep.get('name', '')
                # 找出所有漏洞的修復版本
                fix_versions = []
                for vuln in dep.get('vulns', []):
                    fix_versions.extend(vuln.get('fix_versions', []))

                if fix_versions:
                    # 取最新的修復版本
                    latest_fix = sorted(set(fix_versions), reverse=True)[0]
                    print(f"   pip install --upgrade {name}>={latest_fix}")
                else:
                    print(f"   ⚠️  {name}: 目前無修復版本,建議關注官方更新")
        else:
            print("\n✅ 未發現已知漏洞!")

        print(f"\n✅ pip-audit 掃描完成")
        return True, data

    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失敗: {str(e)}")
        print(f"輸出內容前500字: {result.stdout[:500]}")
        return False, None
    except Exception as e:
        print(f"❌ 處理失敗: {str(e)}")
        return False, None

def calculate_risk_score(bandit_data, pip_audit_data):
    """計算風險評分 (0-100,越低越好)"""
    score = 0

    # Bandit 問題計分
    if bandit_data:
        results = bandit_data.get('results', [])
        high_count = sum(1 for r in results if r.get('issue_severity') == 'HIGH')
        medium_count = sum(1 for r in results if r.get('issue_severity') == 'MEDIUM')
        low_count = sum(1 for r in results if r.get('issue_severity') == 'LOW')

        score += high_count * 25  # 每個高風險 +25
        score += medium_count * 3  # 每個中風險 +3
        score += low_count * 0.5  # 每個低風險 +0.5

    # pip-audit 漏洞計分(只計算有漏洞的套件)
    if pip_audit_data:
        dependencies = pip_audit_data.get('dependencies', [])
        vulnerable_packages = [dep for dep in dependencies if dep.get('vulns', [])]
        for dep in vulnerable_packages:
            vulns = dep.get('vulns', [])
            score += len(vulns) * 10  # 每個依賴漏洞 +10

    return min(int(score), 100)  # 最高100分

def get_risk_level(score):
    """根據評分返回風險等級"""
    if score == 0:
        return "🟢 優秀", "green"
    elif score <= 15:
        return "🟡 良好", "yellow"
    elif score <= 40:
        return "🟠 中等", "orange"
    elif score <= 70:
        return "🔴 需要關注", "red"
    else:
        return "🔥 需要緊急處理", "critical"

def generate_html_report(bandit_data, pip_audit_data, timestamp):
    """生成 HTML 格式的綜合安全報告"""

    # 計算統計資料
    bandit_results = bandit_data.get('results', []) if bandit_data else []
    bandit_high = sum(1 for r in bandit_results if r.get('issue_severity') == 'HIGH')
    bandit_medium = sum(1 for r in bandit_results if r.get('issue_severity') == 'MEDIUM')
    bandit_low = sum(1 for r in bandit_results if r.get('issue_severity') == 'LOW')

    pip_dependencies = pip_audit_data.get('dependencies', []) if pip_audit_data else []
    pip_vulnerable = [dep for dep in pip_dependencies if dep.get('vulns', [])]
    pip_total_vulns = sum(len(dep.get('vulns', [])) for dep in pip_vulnerable)

    risk_score = calculate_risk_score(bandit_data, pip_audit_data)
    risk_level, risk_color = get_risk_level(risk_score)

    # HTML 模板
    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>安全掃描報告 - {timestamp}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f7fa; padding: 20px; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; margin-bottom: 30px; }}
        h2 {{ color: #34495e; margin-top: 40px; margin-bottom: 20px; border-left: 4px solid #3498db; padding-left: 15px; }}
        h3 {{ color: #555; margin-top: 25px; margin-bottom: 15px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 40px; }}
        .card {{ padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .card h3 {{ margin-top: 0; font-size: 14px; text-transform: uppercase; color: #7f8c8d; }}
        .card .value {{ font-size: 36px; font-weight: bold; margin: 10px 0; }}
        .card.green {{ background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); color: #155724; }}
        .card.yellow {{ background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); color: #856404; }}
        .card.orange {{ background: linear-gradient(135deg, #ffe5d0 0%, #ffd4a3 100%); color: #8a4b08; }}
        .card.red {{ background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); color: #721c24; }}
        .card.critical {{ background: linear-gradient(135deg, #f8d7da 0%, #e74c3c 100%); color: #721c24; }}
        .card.blue {{ background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%); color: #0c5460; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #34495e; color: white; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .severity-high {{ color: #dc3545; font-weight: bold; }}
        .severity-medium {{ color: #ffc107; font-weight: bold; }}
        .severity-low {{ color: #28a745; font-weight: bold; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #7f8c8d; font-size: 14px; }}
        .fix-cmd {{ background: #2c3e50; color: #2ecc71; padding: 10px; border-radius: 4px; font-family: 'Courier New', monospace; margin: 5px 0; }}
        .timestamp {{ color: #7f8c8d; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔒 安全掃描報告</h1>
        <p class="timestamp">生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="summary">
            <div class="card {risk_color}">
                <h3>整體風險評分</h3>
                <div class="value">{risk_score}/100</div>
                <p>{risk_level}</p>
            </div>
            <div class="card blue">
                <h3>Bandit 代碼問題</h3>
                <div class="value">{len(bandit_results)}</div>
                <p>🔴 高: {bandit_high} | 🟡 中: {bandit_medium} | 🟢 低: {bandit_low}</p>
            </div>
            <div class="card {'red' if pip_vulnerable else 'green'}">
                <h3>依賴套件漏洞</h3>
                <div class="value">{len(pip_vulnerable)}</div>
                <p>套件數 / 總漏洞: {pip_total_vulns}</p>
            </div>
        </div>

        <h2>📊 Bandit 靜態分析結果</h2>
"""

    # Bandit 結果表格
    if bandit_results:
        severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        sorted_bandit = sorted(bandit_results, key=lambda x: severity_order.get(x.get('issue_severity', 'LOW'), 3))

        html += """
        <table>
            <thead>
                <tr>
                    <th>嚴重性</th>
                    <th>測試ID</th>
                    <th>檔案位置</th>
                    <th>問題描述</th>
                </tr>
            </thead>
            <tbody>
"""
        for issue in sorted_bandit:
            severity = issue.get('issue_severity', 'UNKNOWN')
            test_id = issue.get('test_id', '')
            filename = issue.get('filename', '').replace('\\', '/')
            line_number = issue.get('line_number', '')
            issue_text = issue.get('issue_text', '')
            severity_class = f"severity-{severity.lower()}"

            html += f"""
                <tr>
                    <td class="{severity_class}">{severity}</td>
                    <td>{test_id}</td>
                    <td>{filename}:{line_number}</td>
                    <td>{issue_text}</td>
                </tr>
"""
        html += """
            </tbody>
        </table>
"""
    else:
        html += "<p>✅ 未發現安全問題!</p>"

    # pip-audit 結果
    html += """
        <h2>🔍 pip-audit 依賴漏洞掃描結果</h2>
"""

    if pip_vulnerable:
        html += f"<p>掃描了 {len(pip_dependencies)} 個套件,發現 {len(pip_vulnerable)} 個套件存在 {pip_total_vulns} 個漏洞:</p>"

        for dep in pip_vulnerable:
            name = dep.get('name', '')
            version = dep.get('version', '')
            vulns = dep.get('vulns', [])

            html += f"""
        <h3>📦 {name} (版本 {version}) - {len(vulns)} 個漏洞</h3>
        <table>
            <thead>
                <tr>
                    <th>漏洞ID</th>
                    <th>別名 (CVE)</th>
                    <th>修復版本</th>
                </tr>
            </thead>
            <tbody>
"""
            for vuln in vulns:
                vuln_id = vuln.get('id', '')
                aliases = ', '.join(vuln.get('aliases', []))
                fix_versions = ', '.join(vuln.get('fix_versions', [])) or '尚無修復版本'

                html += f"""
                <tr>
                    <td>{vuln_id}</td>
                    <td>{aliases}</td>
                    <td>{fix_versions}</td>
                </tr>
"""
            html += """
            </tbody>
        </table>
"""
            # 修復建議
            fix_versions = []
            for vuln in vulns:
                fix_versions.extend(vuln.get('fix_versions', []))

            if fix_versions:
                latest_fix = sorted(set(fix_versions), reverse=True)[0]
                html += f'<div class="fix-cmd">$ pip install --upgrade {name}>={latest_fix}</div>'
    else:
        html += "<p>✅ 未發現已知漏洞!</p>"

    # 結尾
    html += f"""
        <div class="footer">
            <p>此報告由 security_check.py 自動生成 | Speech-to-Text 專案安全掃描</p>
            <p>報告時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""

    return html

def main():
    """主函數"""
    # 修復 Windows 命令行編碼問題
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 80)
    print("🔒 Speech-to-Text 專案安全檢測工具 (優化版)")
    print("=" * 80)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("優化: 每個工具只執行一次,生成統一 HTML 報告")

    # 執行 bandit
    bandit_success, bandit_data = run_bandit()

    # 執行 pip-audit
    pip_audit_success, pip_audit_data = run_pip_audit()

    # 生成報告
    print_section("生成綜合報告")

    if bandit_success or pip_audit_success:
        # 儲存 JSON 數據
        os.makedirs("security_reports", exist_ok=True)

        report_data = {
            "timestamp": timestamp,
            "generated_at": datetime.now().isoformat(),
            "bandit": bandit_data,
            "pip_audit": pip_audit_data
        }

        json_report_path = f"security_reports/security_report_{timestamp}.json"
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"✅ JSON 報告已儲存: {json_report_path}")

        # 生成 HTML 報告
        html_content = generate_html_report(bandit_data, pip_audit_data, timestamp)
        html_report_path = f"security_reports/security_report_{timestamp}.html"

        with open(html_report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ HTML 報告已儲存: {html_report_path}")

        # 計算風險評分
        risk_score = calculate_risk_score(bandit_data, pip_audit_data)
        risk_level, _ = get_risk_level(risk_score)

        print_section("檢測完成")
        print(f"\n📊 整體安全評估:")
        print(f"   風險評分: {risk_score}/100")
        print(f"   風險等級: {risk_level}")

        # 統計總問題數
        total_bandit_issues = len(bandit_data.get('results', [])) if bandit_data else 0

        # 統計有漏洞的套件數和總漏洞數
        vulnerable_packages_count = 0
        total_pip_issues = 0
        if pip_audit_data:
            dependencies = pip_audit_data.get('dependencies', [])
            vulnerable_packages = [dep for dep in dependencies if dep.get('vulns', [])]
            vulnerable_packages_count = len(vulnerable_packages)
            total_pip_issues = sum(len(dep.get('vulns', [])) for dep in vulnerable_packages)

        print(f"\n   總計發現:")
        print(f"   • Bandit 代碼問題: {total_bandit_issues}")
        print(f"   • 有漏洞的依賴套件: {vulnerable_packages_count} 個 (共 {total_pip_issues} 個漏洞)")

        print(f"\n📁 詳細報告:")
        print(f"   • HTML 報告: {html_report_path}")
        print(f"   • JSON 數據: {json_report_path}")

        # 提供優先級建議
        if risk_score > 30:
            print("\n🚨 緊急行動建議:")
            action_num = 1
            if bandit_data:
                high_issues = [r for r in bandit_data.get('results', []) if r.get('issue_severity') == 'HIGH']
                if high_issues:
                    print(f"   {action_num}. 立即修復 {len(high_issues)} 個 Bandit 高風險問題")
                    action_num += 1
            if vulnerable_packages_count > 0:
                print(f"   {action_num}. 更新 {vulnerable_packages_count} 個有漏洞的依賴套件")
        elif risk_score > 0:
            print("\n💡 改進建議:")
            if total_bandit_issues > 0:
                print(f"   • 審查並修復 {total_bandit_issues} 個 Bandit 檢測到的問題")
            if vulnerable_packages_count > 0:
                print(f"   • 更新 {vulnerable_packages_count} 個有漏洞的依賴套件")

        print("\n💡 最佳實踐建議:")
        print("  1. 定期執行此腳本 (建議每週一次)")
        print("  2. 在部署前執行一次完整檢測")
        print("  3. 優先修復高風險問題和已知漏洞")
        print("  4. 保持依賴套件為最新穩定版本")
        print("  5. 將安全檢測整合到 CI/CD 流程")

        return 0
    else:
        print("❌ 安全檢測失敗,請確認工具已正確安裝")
        print("\n安裝指令:")
        print("  pip install bandit pip-audit")
        return 1

if __name__ == "__main__":
    sys.exit(main())
